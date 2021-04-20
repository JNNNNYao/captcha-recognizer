import tensorflow as tf
import time

import data
import tools
from config import constants
from transformer import Transformer

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



test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')



def test_accuracy_function(real, pred):
    accuracies = tf.equal(real, pred)

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    accu = tf.reduce_sum(accuracies)/tf.reduce_sum(mask)
    return int(accu == 1)   # incorrect when words not matches exactly



@tf.function()
def evaluation(inp, output):
    combined_mask, dec_padding_mask = tools.create_masks(
            inp, output)   

    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = transformer(inp,
                                                output,
                                                False,
                                                combined_mask,
                                                dec_padding_mask)

    # select the last word from the seq_len dimension
    predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.argmax(predictions, axis=-1)

    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, tf.cast(predicted_id, tf.int32)], axis=-1)
    return output



def main():
    ds, ds_val, tokenizer = data.create_dataset()

    checkpoint_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint(transformer=transformer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    
    start_time = time.time()

    test_accuracy.reset_states()

    max_length=6

    for (batch, (inp, tar)) in enumerate(ds_val):
        start = tokenizer.word_index['<start>']
        output = tf.convert_to_tensor([start])
        output = tf.expand_dims(output, 0)

        for i in range(max_length):
            output = evaluation(inp, output)

        # text = [tokenizer.index_word[id] for id in output[0].numpy()]
        # print(text)
        test_accuracy(test_accuracy_function(tar, output))
        if (batch+1) % 100 == 0:
            print(f'Batch {batch+1} Test accuracy {test_accuracy.result():.4f}')

    print(f'Time taken for testing: {time.time() - start_time:.2f} secs\n')



if __name__ == '__main__':
    try:
        import colored_traceback
        colored_traceback.add_hook()
    except ImportError:
        pass

    main()