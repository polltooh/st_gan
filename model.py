import tensorflow as tf
import save_func as sf
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
                                is_train, self.bsize)

        self.st_data = batch_tensor_list[0]
        self.image_data = batch_tensor_list[1]

    def model(self):
        wd = 0.0004
        leaky_param = 0.01
        with tf.variable_scope("G"):
            with tf.variable_scope("st"):
                sf1 = mf.add_leaky_relu(mf.fully_connected_layer(self.st_data_ph, 100, wd, "fc1"), leaky_param)
                sf2 = mf.add_leaky_relu(mf.fully_connected_layer(sf1, 8 * 8 * 64, wd, "fc2"), leaky_param)
                #sf3 = tf.reshape(sf2, [self.bsize, 8, 8, 64], name = "fc3")
                sf3 = tf.reshape(sf2, [self.bsize, 8, 8, 64], name = "fc3")
                sdeconv1 = mf.add_leaky_relu(mf.deconvolution_2d_layer(sf3, [2, 2, 64, 64], [2,2], 
                                    [self.bsize, 16, 16, 64], "VALID", wd, "deconv1"), leaky_param)
                sdeconv2 = mf.add_leaky_relu(mf.deconvolution_2d_layer(sdeconv1, [2, 2, 64, 64], 
                                    [2,2], [self.bsize, 32, 32, 64], "VALID", wd, "deconv2"), leaky_param)

            with tf.variable_scope("ran"):
                rf2 = mf.add_leaky_relu(mf.fully_connected_layer(self.ran_code_ph, 8 * 8 * 64, wd, "fc2"), leaky_param)
                rf3 = tf.reshape(rf2, [self.bsize, 8, 8, 64], name = "fc3")
                rdeconv1 = mf.add_leaky_relu(mf.deconvolution_2d_layer(rf3, [2, 2, 64, 64], 
                    [2,2], [self.bsize, 16, 16, 64], "VALID", wd, "deconv1"), leaky_param)

                rdeconv2 = mf.add_leaky_relu(mf.deconvolution_2d_layer(rdeconv1, [2, 2, 64, 64], 
                                [2,2], [self.bsize, 32, 32, 64], "VALID", wd, "deconv2"), leaky_param)

            concat = tf.concat(3, [sdeconv2, rdeconv2])
            conv1 = mf.add_leaky_relu(mf.convolution_2d_layer(concat, [3, 3, 128, 256], [1,1], "SAME", wd, "conv1"), leaky_param)

            aconv1 = mf.add_leaky_relu(mf.atrous_convolution_layer(conv1, [3,3, 256, 512], 2, "SAME", wd, "aconv1"), leaky_param)
            aconv2 = mf.add_leaky_relu(mf.atrous_convolution_layer(aconv1, [3,3, 512, 256], 2, "SAME", wd, "aconv2"), leaky_param)

            deconv1 = mf.add_leaky_relu(mf.deconvolution_2d_layer(aconv2, [2, 2, 128, 256], 
                            [2,2], [self.bsize, 64, 64, 128], "VALID", wd, "deconv1"), leaky_param)

            deconv2 = mf.add_leaky_relu(mf.deconvolution_2d_layer(deconv1, [2, 2, 64, 128], 
                            [2,2], [self.bsize, 128, 128, 64], "VALID", wd, "deconv2"), leaky_param)
            
            conv11 = mf.convolution_2d_layer(deconv2, [5, 5, 64, 64], [1,1], "SAME", wd, "conv11")
            self.g_image = mf.convolution_2d_layer(conv11, [1, 1, 64, 1], [1,1], "SAME", wd, "g_image")
            tf.add_to_collection("image_to_write", self.g_image)



        with tf.variable_scope("D"):
            concat_real_fake = tf.concat(3, [self.g_image, self.image_data_ph])
            concat_real_real = tf.concat(3, [tf.random_shuffle(self.image_data_ph), self.image_data_ph])
            # real image first and then fake image
            concat = tf.concat(0, [concat_real_real, concat_real_fake])
            #concat = tf.concat(0, [self.g_image, self.image_data_ph])
            #conv1 = mf.convolution_2d_layer(self.image_data_ph, [5, 5, 2, 64], [2,2], "VALID", wd, "conv1")
            conv1 = mf.add_leaky_relu(mf.convolution_2d_layer(concat, [5, 5, 2, 64], [2,2], "VALID", wd, "conv1"), leaky_param)
            conv2 = mf.add_leaky_relu(mf.convolution_2d_layer(conv1, [5, 5, 64, 128], [2,2], "VALID", wd, "conv2"), leaky_param)
            conv3 = mf.add_leaky_relu(mf.convolution_2d_layer(conv2, [5, 5, 128, 512], [2,2], "VALID", wd, "conv3"), leaky_param)
            conv4 = mf.add_leaky_relu(mf.convolution_2d_layer(conv3, [5, 5, 512, 128], [2,2], "VALID", wd, "conv4"), leaky_param)
            self.fc = mf.fully_connected_layer(conv4, 1, wd, "fc")

        #decode network
        with tf.variable_scope("C"):
            conv1 = mf.add_leaky_relu(mf.convolution_2d_layer(self.g_image, [5, 5, 1, 64], [2,2], "VALID", wd, "conv1"), leaky_param)
            conv2 = mf.add_leaky_relu(mf.convolution_2d_layer(conv1, [5, 5, 64, 128], [2,2], "VALID", wd, "conv2"), leaky_param)
            conv3 = mf.add_leaky_relu(mf.convolution_2d_layer(conv2, [5, 5, 128, 512], [2,2], "VALID", wd, "conv3"), leaky_param)
            conv4 = mf.add_leaky_relu(mf.convolution_2d_layer(conv3, [5, 5, 512, 128], [2,2], "VALID", wd, "conv4"), leaky_param)
            self.st_infer = mf.fully_connected_layer(conv4, self.st_len, wd, "fc")

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

        self.g_loss = tf.reduce_mean(g_lambda * g_total_loss[self.bsize:,:], name = "g_loss")
        tf.add_to_collection("losses", self.g_loss)
        
        self.c_loss = mf.l2_loss(self.st_infer, tf.squeeze(self.st_data_ph), "MEAN", "c_loss")
        tf.add_to_collection("losses", self.c_loss)


    def train(self):
        d_vars = [v for v in tf.trainable_variables() if "D" in v.op.name]
        g_vars = [v for v in tf.trainable_variables() if "G" in v.op.name]
        c_vars = [v for v in tf.trainable_variables() if "C" in v.op.name]

        self.d_optim = tf.train.AdamOptimizer(self.init_learning_rate).minimize(self.d_loss, var_list = d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.init_learning_rate).minimize(self.g_loss, var_list = g_vars)
        self.c_optim = tf.train.AdamOptimizer(self.init_learning_rate).minimize(self.c_loss, var_list = g_vars + c_vars)


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
                _, _, g_image_v, g_loss_v, c_loss_v, summ_v = sess.run([self.g_optim, self.c_optim, self.g_image, 
                                            self.g_loss, self.c_loss, summ], feed_dict = feed_data)
            if i%100 == 0:
                sum_writer.add_summary(summ_v, i)
                print("iter: %d, d_loss: %.2f, g_loss: %.2f, c_loss: %.2f"%(i, d_loss_v, g_loss_v, c_loss_v))
