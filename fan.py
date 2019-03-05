import tensorflow as tf 
import cv2


class Conv3x3(tf.keras.Model):
    def __init__(self, inplanes, outplanes, strides=1, padding='same', bias=False):
        super(Conv3x3, self).__init__(name='Conv3x3%dto%d'%(inplanes, outplanes))
        self.inplanes = inplanes 
        self.outplanes = outplanes 
        self.strides = strides
        self.padding = padding 
        self.bias = bias 

    def call(self, input_tensor, training=False):
        x = tf.layers.Conv2D(
            filters=self.outplanes,
            kernel_size=(3,3),
            strides=self.strides,
            padding=self.padding,
            use_bias = self.bias
        )
        return x(input_tensor)

class Downsample(tf.keras.Model):
    def __init__(self, outplanes):        
        super(Downsample, self).__init__(name='Downsample')
        self.outplanes = outplanes
        
    def call(self, input_tensor, training=False):
        x = tf.layers.batch_normalization(
            inputs=input_tensor, training=training
        )
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(
            inputs=x, 
            filters=self.outplanes,
            kernel_size=(1,1),
            strides=(1,1),
            use_bias=False
        )
        return x    


class ConvBlock(tf.keras.Model):
    
    def __init__(self, inplanes, outplanes):
        super(ConvBlock, self).__init__(name='ConvBlock%dto%d'%(inplanes, outplanes))
        self.inplanes = inplanes
        self.outplanes = outplanes
    
        self.bn1 = tf.layers.BatchNormalization()        
        self.conv1 = Conv3x3(inplanes, int(outplanes/2))
        self.bn2 = tf.layers.BatchNormalization()
        self.conv2 = Conv3x3(int(outplanes/2), int(outplanes/4))
        self.bn3 = tf.layers.BatchNormalization()
        self.conv3 = Conv3x3(int(outplanes/4), int(outplanes/4))
        
        if inplanes != outplanes:                        
            self.downsample = Downsample(outplanes)
        else:
            self.downsample = None 

    
    def call(self, input_tensor, training=False):
        residual = input_tensor

        out1 = self.bn1(input_tensor, training=training)
        out1 = tf.nn.relu(out1)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1, training=training)
        out2 = tf.nn.relu(out2)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2, training=training)
        out3 = tf.nn.relu(out3)
        out3 = self.conv3(out3)

        out3 = tf.concat(axis=3, values=[out1, out2, out3])
        
        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 = out3 + residual        

        return out3 


class Bottleneck(tf.keras.Model):
    def __init__(self, inplanes, outplanes, stride=1, downsample=None):

        super(Bottleneck, self).__init__(name='Bottleneck%dto%d'%(inplanes, outplanes))
        self.conv1 = tf.layers.Conv2D(
            filters=outplanes,
            kernel_size=(1,1),
            use_bias=False
        )
        self.bn1 = tf.layers.BatchNormalization()
        self.conv2 = tf.layers.Conv2D(
            filters=outplanes,
            kernel_size=(3,3),
            strides=(stride, stride),
            padding='same',
            use_bias=False
        )
        self.bn2 = tf.layers.BatchNormalization()
        self.conv3 = tf.layers.Conv2D(
            filters=outplanes * 4,
            kernel_size=(1,1),            
            use_bias=False
        )
        self.bn3 = tf.layers.BatchNormalization()
        self.downsample = downsample
        self.stride = stride 


    def call(self, input_tensor, training=False):
        residual = input_tensor

        out = self.conv1.apply(input_tensor)
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)

        out = self.conv2.apply(out)
        out = self.bn2(out, training=training)
        out = tf.nn.relu(out)

        out = self.conv3.apply(out)
        out = self.bn3(out, training=training)

        if self.downsample is not None:
            residual = self.downsample(input_tensor)

        out = out + residual
        out = tf.nn.relu(out)

        return out 


class HourGlass(tf.keras.Model):

    def __init__(self, num_modules, depth, num_features):

        super(HourGlass, self).__init__(name="hourglass")
        self.num_modules = num_modules
        self.depth = depth 
        self.features = num_features
        
        self.layers_dict = {}
        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.layers_dict['b1_'+str(level)] = ConvBlock(self.features, self.features)
        self.layers_dict['b2_'+str(level)] = ConvBlock(self.features, self.features)
        
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.layers_dict['b2_plus_'+str(level)] = ConvBlock(self.features, self.features)

        self.layers_dict['b3_'+str(level)] = ConvBlock(self.features, self.features)


    def _call(self, level, input_tensor, training=False):
        # Upper branch
        up1 = input_tensor
        up1 = self.layers_dict['b1_'+str(level)](up1)

        # Lower branch
        low1 = tf.layers.average_pooling2d(input_tensor,2,2)
        low1 = self.layers_dict['b2_'+str(level)](low1)

        if level > 1:
            low2 = self._call(level - 1, low1, training)
        else:
            low2 = low1 
            low2 = self.layers_dict['b2_plus_'+str(level)](low2)

        low3 = low2 
        low3 = self.layers_dict['b3_'+str(level)](low3)

        up2 = tf.keras.layers.UpSampling2D()(low3)

        return up1 + up2 


    def call(self, input_tensor, training=False):
        return self._call(self.depth, input_tensor, training)
        

class FAN(tf.keras.Model):

    def __init__(self, num_modules=1):
        super(FAN, self).__init__()
        self.num_modules = num_modules

        # base
        self.conv1 = tf.layers.Conv2D(64,7,2,'same')
        self.bn1 = tf.layers.BatchNormalization()
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # stacking
        self.layers_dict = {}
        for hg_module in range(self.num_modules):
            self.layers_dict['m'+str(hg_module)] = HourGlass(1, 4, 256)
            self.layers_dict['top_m_'+str(hg_module)] = ConvBlock(256, 256)
            self.layers_dict['conv_last'+str(hg_module)] = tf.layers.Conv2D(256, 1, strides=1, padding='valid')
            self.layers_dict['bn_end'+str(hg_module)] = tf.layers.BatchNormalization()
            self.layers_dict['l'+str(hg_module)] = tf.layers.Conv2D(68,1,strides=1,padding='valid')

            if hg_module < self.num_modules - 1:
                self.layers_dict['b1'+str(hg_module)] = tf.layers.Conv2D(256, 1, strides=1, padding='valid')
                self.layers_dict['a1'+str(hg_module)] = tf.layers.Conv2D(256, 1, strides=1, padding='valid')

    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = tf.layers.average_pooling2d(x, pool_size=2, strides=2)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x
        outputs = []
        for i in range(self.num_modules):
            hg = self.layers_dict['m'+str(i)](previous)

            ll = hg 
            ll = self.layers_dict['top_m_'+str(i)](ll)

            ll = tf.nn.relu(self.layers_dict['bn_end'+str(i)](
                    self.layers_dict['conv_last'+str(i)](ll), training))

            # heatmaps
            tmp_out = self.layers_dict['l' + str(i)](ll)
            outputs.append(tmp_out)

            if i < self.num_modules - 1:
                ll = self.layers_dict['b1'+str(i)](ll)
                tmp_out_ = self.layers_dict['a1'+str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        return outputs