import tensorflow as tf
import time

import data
import tools
from config import constants
from transformer import Transformer

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=5000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

        

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')



learning_rate = CustomSchedule(constants['d_model'])
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
        epsilon=1e-9)



train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')



transformer = Transformer(
        CNN_layers = constants['CNN_layers'],
        CNN_depth = constants['CNN_depth'],
        decoder_layers=constants['decoder_layers'],
        d_model=constants['d_model'],
        num_heads=constants['num_heads'],
        dff=constants['dff'],
        target_vocab_size=29,
        pe_target=constants['pe_target'],
        rate=constants['dropout_rate'])



def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)



def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2, output_type=tf.int32))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)



@tf.function()
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    combined_mask, dec_padding_mask = tools.create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                    True,
                                    combined_mask,
                                    dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))



def main():
    ds, ds_val, tokenizer = data.create_dataset()

    checkpoint_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint(transformer=transformer,
                            optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    
    for epoch in range(constants['EPOCHS']):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(ds):
            train_step(inp, tar)

            if batch % 10 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
            
            if (batch + 1) % 1000 == 0:
                ckpt_save_path = ckpt_manager.save()
                print(f'Saving checkpoint for batch {batch+1} at {ckpt_save_path}')

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')



if __name__ == '__main__':
    try:
        import colored_traceback
        colored_traceback.add_hook()
    except ImportError:
        pass

    main()

