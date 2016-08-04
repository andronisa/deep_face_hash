import h5py
from keras.models import Sequential
from keras import backend as K


class DeepFaceHashSequential(Sequential):
    def load_weights(self, filepath, excluded_layers=list()):
        print("Custom Weights Loader")

        '''Load all layer weights from a HDF5 save file.
        '''
        f = h5py.File(filepath, mode='r')

        if hasattr(self, 'flattened_layers'):
            # support for legacy Sequential/Merge behavior
            flattened_layers = self.flattened_layers
        else:
            flattened_layers = self.layers

        if 'nb_layers' in f.attrs:
            # legacy format
            nb_layers = f.attrs['nb_layers']
            if nb_layers != len(flattened_layers):
                raise Exception('You are trying to load a weight file '
                                'containing ' + str(nb_layers) +
                                ' layers into a model with ' +
                                str(len(flattened_layers)) + '.')

            for k in range(nb_layers):
                g = f['layer_{}'.format(k)]
                weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
                flattened_layers[k].set_weights(weights)
        else:
            # new file format
            layer_names = [n.decode('utf8') for n in f.attrs['layer_names'] if n not in excluded_layers]
            # we batch weight value assignments in a single backend call
            # which provides a speedup in TensorFlow.
            weight_value_tuples = []
            for k, name in enumerate(layer_names):
                g = f[name]
                weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
                if len(weight_names):
                    weight_values = [g[weight_name] for weight_name in weight_names]

                    layer = flattened_layers[k]
                    symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
                    if len(weight_values) != len(symbolic_weights):
                        raise Exception('Layer #' + str(k) +
                                        ' (named "' + layer.name +
                                        '" in the current model) was found to '
                                        'correspond to layer ' + name +
                                        ' in the save file. '
                                        'However the new layer ' + layer.name +
                                        ' expects ' + str(len(symbolic_weights)) +
                                        ' weights, but the saved weights have ' +
                                        str(len(weight_values)) +
                                        ' elements.')
                    weight_value_tuples += zip(symbolic_weights, weight_values)
            K.batch_set_value(weight_value_tuples)
        f.close()
