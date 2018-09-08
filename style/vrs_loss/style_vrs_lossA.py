
import tensorflow as tf
import kaleido as kld

##### VRS
class vrs( kld.tf.vrs ):

    ### BUILD
    def build( self , vrsnet , vgg , content , style ):

        content = kld.aux.order( content )
        style = kld.aux.order( style )

        self.yc = kld.tf.plchf( [ None , None , None , 3 ] , 'content' )
        self.ys = kld.tf.plchf( [ None , None , None , 3 ] , 'style'   )
        self.yh = vrsnet( 'Loss' , self.yc ).output * 255.0

        content_layers = vgg.build( self.yc * 255.0 , 'content' )
        style_layers   = vgg.build( self.ys * 255.0 , 'style'   )
        self.Fs        = vgg.build( self.yh         , 'mixed'   )

        self.Ss = {}
        for id in style:
            self.Ss[ id ] = style_layers[ id ]

        self.Cs = {}
        for id in content:
            self.Cs[ id ] = content_layers[ id ]

        loss_content , loss_style = 0 , 0
        for id in self.Fs:

            if id in content:

                F = self.Fs[ id ]
                C = self.Cs[ id ]

                b , h , w , d = kld.tf.shape( F )
                bhwd = kld.tf.ops.toFloat( b * h * w * d )
                wgt = content[ id ]

                loss_content += wgt * 2 * tf.nn.l2_loss( F - C ) / bhwd

            if id in style:

                F = kld.tf.ops.gram_matrix( self.Fs[ id ] )
                S = kld.tf.ops.gram_matrix( self.Ss[ id ] )

                b , d1 , d2 = kld.tf.shape( F )
                bd1d2 = kld.tf.ops.toFloat( b * d1 * d2 )
                wgt = style[ id ]

                loss_style += wgt * 2 * tf.nn.l2_loss( F - S ) / bd1d2

        loss_totvar = kld.tf.ops.total_variation_loss( self.yh )

        self.content = self.args.wgt_content * loss_content
        self.style   = self.args.wgt_style   * loss_style
        self.totvar  = self.args.wgt_totvar  * loss_totvar

        self.full = self.content + self.style + self.totvar
        self.all = [ self.content , self.style , self.totvar , self.full ]




