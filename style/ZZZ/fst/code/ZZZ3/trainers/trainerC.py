
import numpy as np
import tensorflow as tf
import kaleido as kld
from trainers.trainer3 import TrainerBase

##### TRAINER
class Trainer( TrainerBase ):

    ### BUILD
    def build( self , args ):
        self.preBuild()

        self.ys  = kld.plchf( [        None , None , 3 ] , 'style'   )
        self.yc1 = kld.plchf( [ None , None , None , 3 ] , 'content' )
        self.yc2 = kld.plchf( [ None , None , None , 3 ] , 'content' )

        with tf.variable_scope( '' , reuse = tf.AUTO_REUSE ):
            args.net.type = 'calc'
            args.net.build( self.yc1 )
            args.net.type = 'eval'
            self.yh = args.net.build( self.yc2 )

        self.ysi = tf.expand_dims( self.ys , 0 )

        style_layers   = args.vgg_net.feed_forward( self.ysi  , 'style'   )
        content_layers = args.vgg_net.feed_forward( self.yc2  , 'content' )
        self.Fs        = args.vgg_net.feed_forward( self.yh   , 'mixed'   )

        self.Ss = {}
        for id in self.style_layers:
            self.Ss[ id ] = style_layers[ id ]

        self.Cs = {}
        for id in self.content_layers:
            self.Cs[ id ] = content_layers[ id ]

        L_style , L_content = 0 , 0
        for id in self.Fs:

            if id in self.style_layers:

                F = kld.gram_matrix( self.Fs[ id ] )
                S = kld.gram_matrix( self.Ss[ id ] )

                b , d1 , d2 = kld.get_shape( F )
                bd1d2 = kld.toFloat( b * d1 * d2 )
                wgt = self.style_layers[ id ]

                L_style += wgt * 2 * tf.nn.l2_loss( F - S ) / bd1d2

            if id in self.content_layers:

                F = self.Fs[ id ]
                C = self.Cs[ id ]

                b , h , w , d = kld.get_shape( F )
                bhwd = kld.toFloat( b * h * w * d )
                wgt = self.content_layers[ id ]

                L_content += wgt * 2 * tf.nn.l2_loss( F - C ) / bhwd

        L_totvar = kld.total_variation_loss( self.yh )

        self.L_style   = args.wgt_style   * L_style
        self.L_content = args.wgt_content * L_content
        self.L_totvar  = args.wgt_totvar  * L_totvar
        self.L_full    = self.L_style + self.L_content + self.L_totvar

