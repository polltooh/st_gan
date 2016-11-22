import tensorflow as tf
import save_func as sf
import cv2
import tensor_data
import data_class
import model_func as mf
import os
import numpy as np
import read_proto as rp


class STGan(object):
    def __init__(self, model_params):

        self.bsize = model_params["batch_size"]
        self.st_len = model_params["st_len"]
        self.code_len = model_params["code_len"]
        self.init_learning_rate = model_params["init_learning_rate"]
        self.max_training_iter = model_params["max_training_iter"]
        self.iheight = model_params["iheight"]
        self.iwidth = model_params["iwidth"]
        self.g_iter = model_params["g_iter"]
        self.d_iter = model_params["d_iter"]
        self.train_log_dir = model_params["train_log_dir"]
        self.restore_model = model_params["restore_model"]
        self.model_dir = model_params["model_dir"]

        #self.test_shape()
        self.st_data_ph = tf.placeholder(tf.float32, 
                    shape = (self.bsize, model_params["st_len"], 1), name = 'st_data')

        self.image_data_ph = tf.placeholder(tf.float32, 
                    shape = (self.bsize, model_params["iheight"], 
                        model_params["iwidth"], 1), name = 'img_data')
        self.ran_code_ph = tf.placeholder(tf.float32,
                    shape = (self.bsize, model_params["code_len"]), name = 'random_code')

        if not os.path.exists(self.train_log_dir):
            os.makedirs(self.train_log_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.model()
        self.loss()
        self.train()
	self.data_load(model_params["file_name"])

    def test_shape(self):
        input_tensor = tf.constant(1, np.float32, (self.bsize, self.iheight, self.iwidth, 1))
        conv1 = mf.convolution_2d_layer(input_tensor, [4, 4, 1, 1], [2,2], "VALID", 0.0 , "conv1")
        conv2 = mf.convolution_2d_layer(conv1, [4, 4, 1, 1], [2,2], "VALID", 0.0 , "conv2")
        print(conv1.get_shape())
        print(conv2.get_shape())
        exit(1)

    def data_load(self, file_name):
        is_train = True

        st_data = data_class.DataClass(tf.constant([], tf.string))
        st_data.decode_class = data_class.BINClass((self.st_len,1))

        image_data = data_class.DataClass(tf.constant([], tf.string))
        image_data.decode_class = data_class.JPGClass((self.iheight, self.iwidth), 1, 0)

        tensor_list = [st_data] + [image_data]

        file_queue = tensor_data.file_queue(file_name, is_train)
        batch_tensor_list = tensor_data.file_queue_to_batch_data(file_queue, tensor_list, 
                                is_train, self.bsize, False)

        self.st_data = batch_tensor_list[0]
        self.image_data = batch_tensor_list[1]

    def model(self):
        wd = 0.0004
        leaky_param = 0.01
        with tf.variable_scope("G"):
            rf2 = mf.add_leaky_relu(mf.fully_connected_layer(self.ran_code_ph, 8 * 8 * 64, wd, "fc2"), leaky_param)
            rf3 = tf.reshape(rf2, [self.bsize, 8, 8, 64], name = "fc3")
            rdeconv1 = mf.add_leaky_relu(mf.deconvolution_2d_layer(rf3, [2, 2, 128, 64], 
                [2,2], [self.bsize, 16, 16, 128], "VALID", wd, "deconv1"), leaky_param)

            rdeconv2 = mf.add_leaky_relu(mf.deconvolution_2d_layer(rdeconv1, [2, 2, 256, 128], 
                            [2,2], [self.bsize, 32, 32, 256], "VALID", wd, "deconv2"), leaky_param)
            deconv1 = mf.add_leaky_relu(mf.deconvolution_2d_layer(rdeconv2, [2, 2, 512, 256], 
                            [2,2], [self.bsize, 64, 64, 512], "VALID", wd, "deconv3"), leaky_param)

            deconv2 = mf.add_leaky_relu(mf.deconvolution_2d_layer(deconv1, [2, 2, 512, 512], 
                            [2,2], [self.bsize, 128, 128, 512], "VALID", wd, "deconv4"), leaky_param)
            conv1 = mf.convolution_2d_layer(deconv2, [1, 1, 512, 1], [1,1], "SAME", wd, "conv1")

            self.g_image = tf.sigmoid(conv1, "g_image")

            tf.add_to_collection("image_to_write", self.g_image)
            tf.add_to_collection("image_to_write", self.image_data_ph)

        with tf.variable_scope("D"):
            concat = tf.concat(0, [self.g_image, self.image_data_ph])
            #conv1 = mf.convolution_2d_layer(self.image_data_ph, [5, 5, 2, 64], [2,2], "VALID", wd, "conv1")
            conv1 = mf.add_leaky_relu(mf.convolution_2d_layer(concat, [3, 3, 1, 16], [2,2], "SAME", wd, "conv1"), leaky_param)
            conv1_maxpool = mf.maxpool_2d_layer(conv1, [2,2], [2,2], "maxpool1")

            conv2 = mf.add_leaky_relu(mf.convolution_2d_layer(conv1_maxpool, [3, 3, 16, 32], [2,2], "SAME", wd, "conv2"), leaky_param)
            conv2_maxpool = mf.maxpool_2d_layer(conv2, [2,2], [2,2], "maxpool2")

            conv3 = mf.add_leaky_relu(mf.convolution_2d_layer(conv2, [3, 3, 32, 16], [2,2], "SAME", wd, "conv3"), leaky_param)
            conv3_maxpool = mf.maxpool_2d_layer(conv3, [2,2], [2,2], "maxpool3")

            conv4 = mf.add_leaky_relu(mf.convolution_2d_layer(conv3_maxpool, [3, 3, 16, 1], [2,2], "SAME", wd, "conv4"), leaky_param)
            conv4_maxpool = mf.maxpool_2d_layer(conv4, [2,2], [2,2], "maxpool4")
            self.fc = mf.fully_connected_layer(conv4_maxpool, 1, wd, "fc")

        #decode network
        #with tf.variable_scope("C"):
        #    conv1 = mf.add_leaky_relu(mf.convolution_2d_layer(self.g_image, [5, 5, 1, 64], [2,2], "VALID", wd, "conv1"), leaky_param)
        #    conv2 = mf.add_leaky_relu(mf.convolution_2d_layer(conv1, [5, 5, 64, 128], [2,2], "VALID", wd, "conv2"), leaky_param)
        #    conv3 = mf.add_leaky_relu(mf.convolution_2d_layer(conv2, [5, 5, 128, 512], [2,2], "VALID", wd, "conv3"), leaky_param)
        #    conv4 = mf.add_leaky_relu(mf.convolution_2d_layer(conv3, [5, 5, 512, 128], [2,2], "VALID", wd, "conv4"), leaky_param)
        #    self.st_infer = mf.fully_connected_layer(conv4, self.st_len, wd, "fc")

    def loss(self):
        real_label = tf.constant(1, dtype = tf.float32, shape = (self.bsize, 1))
        fake_label = tf.constant(0, dtype = tf.float32, shape = (self.bsize, 1))

        fake_real_label = tf.concat(0, [fake_label, real_label])
        self.d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.fc, fake_real_label), name = "d_loss")
        tf.add_to_collection("losses", self.d_loss)
        # switch label order
        real_fake_label = tf.concat(0, [real_label, fake_label])
        g_total_loss = tf.nn.sigmoid_cross_entropy_with_logits(self.fc, real_fake_label)
        g_lambda = 1.0

        self.g_loss = tf.reduce_mean(g_lambda * g_total_loss[:self.bsize,:], name = "g_loss")
        tf.add_to_collection("losses", self.g_loss)
        
        #self.c_loss = mf.l2_loss(self.st_infer, tf.squeeze(self.st_data_ph), "MEAN", "c_loss")
        #tf.add_to_collection("losses", self.c_loss)


    def train(self):
        d_vars = [v for v in tf.trainable_variables() if "D" in v.op.name]
        g_vars = [v for v in tf.trainable_variables() if "G" in v.op.name]
        #c_vars = [v for v in tf.trainable_variables() if "C" in v.op.name]

        self.d_optim = tf.train.AdamOptimizer(self.init_learning_rate).minimize(self.d_loss, var_list = d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.init_learning_rate).minimize(self.g_loss, var_list = g_vars)
        #self.c_optim = tf.train.AdamOptimizer(self.init_learning_rate).minimize(self.c_loss, var_list = g_vars + c_vars)


    def mainloop(self):
        sess = tf.Session()
        
        sf.add_train_var()
        sf.add_loss()
        sf.add_image("image_to_write")
        sum_writer = tf.train.SummaryWriter(self.train_log_dir, sess.graph)

        saver = tf.train.Saver()
        summ = tf.merge_all_summaries()

        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        if self.restore_model:
            sf.restore_model(sess, saver, self.model_dir)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord, sess = sess)

        for i in xrange(self.max_training_iter):
            train_st_data_v, train_image_data_v = sess.run([self.st_data, self.image_data])
            #train_image_data_v = np.random.uniform(-1, 1, (self.bsize, self.iheight, self.iwidth, 1))
            #train_st_data_v = np.random.uniform(-1, 1, (self.bsize, self.st_len))

            ran_code = np.random.uniform(-1, 1, size = (self.bsize, 100))

            feed_data = {self.image_data_ph: train_image_data_v,
                                            self.st_data_ph: train_st_data_v,
                                            self.ran_code_ph:ran_code}

            for di in xrange(self.d_iter):
                _, d_loss_v = sess.run([self.d_optim, self.d_loss], feed_dict = feed_data)
            for gi in xrange(self.g_iter):
                #_, _, g_image_v, g_loss_v, c_loss_v, summ_v = sess.run([self.g_optim, self.c_optim, self.g_image, 
                #                            self.g_loss, self.c_loss, summ], feed_dict = feed_data)
                _, g_image_v, g_loss_v, summ_v = sess.run([self.g_optim, self.g_image, 
                                            self.g_loss, summ], feed_dict = feed_data)

            if i%100 == 0:
                sum_writer.add_summary(summ_v, i)
                print("iter: %d, d_loss: %.3f, g_loss: %.3f"%(i, d_loss_v, g_loss_v))

            if i != 0 and (i %1000 == 0 or i = FLAGS.max_training_iter - 1):
                sf.save_model(sess, saver, FLAGS.model_dir, i)
