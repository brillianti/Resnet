import tensorflow as tf
import numpy as np
import os
import resnet_model
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


#Construct the filenames that include the train cifar10 images
folderPath = 'cifar-10-batches-bin/'
filenames = [os.path.join(folderPath, 'data_batch_%d.bin' % i) for i in range(1,6)]

#Define the parameters of the cifar10 image
imageWidth = 32
imageHeight = 32
imageDepth = 3
label_bytes = 1

#Define the train and test batch size
batch_size = 100
test_batch_size = 100
valid_batch_size = 100

#Calulate the per image bytes and record bytes
image_bytes = imageWidth * imageHeight * imageDepth
record_bytes = label_bytes + image_bytes

#Construct the dataset to read the train images
dataset = tf.data.FixedLengthRecordDataset(filenames, record_bytes)
dataset = dataset.shuffle(50000)

#Get the first 45000 records as train dataset records
train_dataset = dataset.take(45000)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.repeat(300)
iterator = train_dataset.make_initializable_iterator()

#Get the remain 5000 records as valid dataset records
valid_dataset = dataset.skip(45000)
valid_dataset = valid_dataset.batch(valid_batch_size)
validiterator = valid_dataset.make_initializable_iterator()

#Construct the dataset to read the test images
testfilename = os.path.join(folderPath, 'test_batch.bin')
testdataset = tf.data.FixedLengthRecordDataset(testfilename, record_bytes)
testdataset = testdataset.batch(test_batch_size)
testiterator = testdataset.make_initializable_iterator()

#Decode the train records from the iterator
record = iterator.get_next()
record_decoded_bytes = tf.decode_raw(record, tf.uint8)

#Get the labels from the records
record_labels = tf.slice(record_decoded_bytes, [0, 0], [batch_size, 1])
record_labels = tf.cast(record_labels, tf.int32)

#Get the images from the records
record_images = tf.slice(record_decoded_bytes, [0, 1], [batch_size, image_bytes])
record_images = tf.reshape(record_images, [batch_size, imageDepth, imageHeight, imageWidth])
record_images = tf.transpose(record_images, [0, 2, 3, 1])
record_images = tf.cast(record_images, tf.float32)

#Decode the records from the valid iterator
validrecord = validiterator.get_next()
validrecord_decoded_bytes = tf.decode_raw(validrecord, tf.uint8)

#Get the labels from the records
validrecord_labels = tf.slice(validrecord_decoded_bytes, [0, 0], [valid_batch_size, 1])
validrecord_labels = tf.cast(validrecord_labels, tf.int32)
validrecord_labels = tf.reshape(validrecord_labels, [-1])

#Get the images from the records
validrecord_images = tf.slice(validrecord_decoded_bytes, [0, 1], [valid_batch_size, image_bytes])
validrecord_images = tf.cast(validrecord_images, tf.float32)
validrecord_images = tf.reshape(validrecord_images,
                               [valid_batch_size, imageDepth, imageHeight, imageWidth])
validrecord_images = tf.transpose(validrecord_images, [0, 2, 3, 1])

#Decode the test records from the iterator
testrecord = testiterator.get_next()
testrecord_decoded_bytes = tf.decode_raw(testrecord, tf.uint8)

#Get the labels from the records
testrecord_labels = tf.slice(testrecord_decoded_bytes, [0, 0], [test_batch_size, 1])
testrecord_labels = tf.cast(testrecord_labels, tf.int32)
testrecord_labels = tf.reshape(testrecord_labels, [-1])

#Get the images from the records
testrecord_images = tf.slice(testrecord_decoded_bytes, [0, 1], [test_batch_size, image_bytes])
testrecord_images = tf.cast(testrecord_images, tf.float32)
testrecord_images = tf.reshape(testrecord_images,
                               [test_batch_size, imageDepth, imageHeight, imageWidth])
testrecord_images = tf.transpose(testrecord_images, [0, 2, 3, 1])

#Random crop the images after pad each side with 4 pixels
distorted_images = tf.image.resize_image_with_crop_or_pad(record_images,
                                                          imageHeight+8, imageWidth+8)
distorted_images = tf.random_crop(distorted_images, size = [batch_size, imageHeight, imageHeight, 3])

#Unstack the images as the follow up operation are on single train image
distorted_images = tf.unstack(distorted_images)
for i in range(len(distorted_images)):
    distorted_images[i] = tf.image.random_flip_left_right(distorted_images[i])
    distorted_images[i] = tf.image.random_brightness(distorted_images[i], max_delta=63)
    distorted_images[i] = tf.image.random_contrast(distorted_images[i], lower=0.2, upper=1.8)
    distorted_images[i] = tf.image.per_image_standardization(distorted_images[i])

#Stack the images
distorted_images = tf.stack(distorted_images)

#transpose to set the channel first
distorted_images = tf.transpose(distorted_images, perm=[0, 3, 1, 2])

#Unstack the images as the follow up operation are on single image
validrecord_images = tf.unstack(validrecord_images)
for i in range(len(validrecord_images)):
    validrecord_images[i] = tf.image.per_image_standardization(validrecord_images[i])

#Stack the images
validrecord_images = tf.stack(validrecord_images)

#transpose to set the channel first
validrecord_images = tf.transpose(validrecord_images, perm=[0, 3, 1, 2])

#Unstack the images as the follow up operation are on single image
testrecord_images = tf.unstack(testrecord_images)
for i in range(len(testrecord_images)):
    testrecord_images[i] = tf.image.per_image_standardization(testrecord_images[i])

#Stack the images
testrecord_images = tf.stack(testrecord_images)

#transpose to set the channel first
testrecord_images = tf.transpose(testrecord_images, perm=[0, 3, 1, 2])

global_step = tf.Variable(0, trainable=False)
boundaries = [10000, 15000, 20000, 25000]
values = [0.1, 0.05, 0.01, 0.005, 0.001]
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
weight_decay = 2e-4
filters = 16  #the first resnet block filter number
n = 5  #the basic resnet block number, total network layers are 6n+2
ver = 2   #the resnet block version

#Get the inference logits by the model
result = resnet_model.inference(distorted_images, True, filters, n, ver)

#Calculate the cross entropy loss
cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=record_labels, logits=result)
cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

#Add the l2 weights to the loss
#Add weight decay to the loss.
l2_loss = weight_decay * tf.add_n(
    # loss is computed using fp32 for numerical stability.
    [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
tf.summary.scalar('l2_loss', l2_loss)
loss = cross_entropy_mean + l2_loss

#Define the optimizer
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)

#Relate to the batch normalization
update_ops = tf.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    opt_op = optimizer.minimize(loss, global_step)

valid_accuracy = tf.placeholder(tf.float32)
test_accuracy = tf.placeholder(tf.float32)
tf.summary.scalar("valid_accuracy", valid_accuracy)
tf.summary.scalar("test_accuracy", test_accuracy)
tf.summary.scalar("learning_rate", learning_rate)

validresult = tf.argmax(resnet_model.inference(validrecord_images, False, filters, n, ver), axis=1)
testresult = tf.argmax(resnet_model.inference(testrecord_images, False, filters, n, ver), axis=1)

#Create the session and run the graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(iterator.initializer)

#Merge all the summary and write
summary_op = tf.compat.v1.summary.merge_all()
train_filewriter = tf.compat.v1.summary.FileWriter('train/', sess.graph)

step = 0
while(True):
    try:
        lossValue, lr, _ = sess.run([loss, learning_rate, opt_op])
        if step % 100 == 0:
            print("step %i: Learning_rate: %f Loss: %f" % (step, lr, lossValue))
        if step % 1000 == 0:
            saver = tf.compat.v1.train.Saver()  #这句话很重要!!!
          #  saver.save(sess,'model/my-model', global_step=step)
            checkpoint_filepath = 'model/train.ckpt'
            saver.save(sess, checkpoint_filepath,global_step=step)
            saver.restore(sess, checkpoint_filepath) #continue training
            truepredictNum = 0
            sess.run([testiterator.initializer, validiterator.initializer])
            accuracy1 = 0.0
            accuracy2 = 0.0
            while(True):
                try:
                    predictValue, testValue = sess.run([validresult, validrecord_labels])
                    truepredictNum += np.sum(predictValue==testValue)
                except tf.errors.OutOfRangeError:
                    print("valid correct num: %i" % (truepredictNum))
                    accuracy1 = truepredictNum / 5000.0
                    break
            truepredictNum = 0
            while(True):
                try:
                    predictValue, testValue = sess.run([testresult, testrecord_labels])
                    truepredictNum += np.sum(predictValue==testValue)
                except tf.errors.OutOfRangeError:
                    print("test correct num: %i" % (truepredictNum))
                    accuracy2 = truepredictNum / 10000.0
                    break
            summary = sess.run(summary_op, feed_dict={valid_accuracy: accuracy1, test_accuracy: accuracy2})
            train_filewriter.add_summary(summary, step)
        step += 1
    except tf.errors.OutOfRangeError:
        break
