import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
# These imports are required to load operators' definition.
import tensorflow_text as tf_text
import sentencepiece as spm
from tensorflow.lite.python import interpreter


TOK_MODEL = "./toktok.model"

TOKENIZER = open(TOK_MODEL, "rb").read()
input_tok = tf_text.SentencepieceTokenizer(TOKENIZER, add_bos=True, add_eos=True)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
modelDir = "GPTs/"

vocabSize = 100000
maxlen = 768
embed_dims = 256
numHeads = 6
feedForwardDims = 768
padId = 0
N = 8
dropoutRate = 0.4
batchSize = 10


@keras.saving.register_keras_serializable(package='GPTembedding')
class GPTembedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    
    def call(self, x):
        maxlen = x.shape[-1]
        position = tf.range(start=0, limit=maxlen, delta=1, dtype=tf.float32)

        # cos = tf.math.sin(position[0::2])
        # sin = tf.math.cos(position[1::2])

        # angle_rads = tf.zeros(maxlen)
        # print(angle_rads)
        # angle_rads[0::2] = sin
        # angle_rads[1::2] = cos

        # position = tf.constant(angle_rads)
        positions = self.pos_emb(position)
        x = self.token_emb(x)
        # print(tf.shape(x), tf.shape(positions))
        return x + positions
@keras.saving.register_keras_serializable(package='GPTblock')
class GPTblock(layers.Layer):
    def __init__(self, embed_dims, num_heads, ffn_dims, dropOut_rate):
        super().__init__()

        self.att = layers.MultiHeadAttention(num_heads, embed_dims, dropout=dropOut_rate)
        self.ffn = keras.Sequential([
            layers.Dense(ffn_dims, activation="gelu"),
            layers.Dense(embed_dims)
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.drop1 = layers.Dropout(dropOut_rate)
        self.drop2 = layers.Dropout(dropOut_rate)
        
    def CausalityMask(self, bs, nds, dtype):
        i = tf.range(nds)[:, None]
        # [0..nb] = [[0]..[nb]]

        j = tf.range(nds)
        # [0..nb]

        m = i >= j - nds+nds
        # [0..nb] + [[0]..[nb]]
        # Result shape is (nb, nb, 1) 
        mask = tf.cast(m, dtype)
        # The tf.cast converts m.type to dtype

        mask = tf.reshape(mask, [1, nds, nds])
        # The tf.reshape converts Mask.shape to [1, nb, ns]
        
        mult = tf.concat(
            [tf.expand_dims(bs, -1), tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)
    
    def call(self, x):
        input_shape = tf.shape(x)
        bs = input_shape[0]
        seq_len = input_shape[1]
        attMask = self.CausalityMask(bs, seq_len, tf.bool)
        att = self.att(x,x, attention_mask = attMask)
        nor1 = self.norm1(x+self.drop1(att))
        ffn = self.ffn(nor1)
        return self.norm2(nor1+self.drop2(ffn)) 
    
@keras.saving.register_keras_serializable(package='GPT')
class GPT(keras.Model):
    def __init__(
        self,
        embed_dim=64,
        num_head=12,
        num_feed_forward=4096,
        maxlen=2048,
        num_layers_dec=1,
        vocab=10,
        dropout_rate = 0.4,
        tokenizer_path = TOK_MODEL
    ):
        super().__init__()
        print("set Model")
        self.tokenizer = spm.SentencePieceProcessor(tokenizer_path)
        self.loss_metric = keras.metrics.Mean(name="loss")
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = maxlen
        self.num_classes = vocab

        self.dec_input = GPTembedding(
            vocab_size=vocab, maxlen=maxlen, embed_dim=embed_dim
        )
        self.gptLayer = []
        for i in range(num_layers_dec):
            self.gptLayer.append(GPTblock(embed_dim, num_head, num_feed_forward, dropout_rate))
        self.classifier = layers.Dense(vocab)

    def decode(self, target):
        y = self.dec_input(target)
        for i in self.gptLayer:
            y = i(y)
        return y

    def call(self, inputs):
        source = inputs
        y = self.decode(source)
        return self.classifier(y)

    @property
    def metrics(self):
        return [self.loss_metric]
 
    def train_step(self, batch):
        # source = batch[:, 0]
        # target = batch[:, 1]
        source = batch["source"]
        target = batch["target"]

        # print(source)
        with tf.GradientTape() as tape: 
            preds = self(source)
            mask = tf.math.logical_not(tf.math.equal(target, -1))
            loss = self.compiled_loss(target, preds, sample_weight=mask)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def test_step(self, batch):
        # source = batch[:, 0]
        # target = batch[:, 1]

        source = batch["source"]
        target = batch["target"]

        preds = self(source)
        mask = tf.math.logical_not(tf.math.equal(target, -1))
        loss = self.compiled_loss(target, preds, sample_weight=mask)
        self.loss_metric.update_state(loss)
        return {"val_loss": self.loss_metric.result()}

def Model(N = 2):
    model = GPT(
        embed_dim = embed_dims,
        num_head = numHeads,
        num_feed_forward = feedForwardDims,
        maxlen = maxlen,
        num_layers_dec=N,
        vocab = vocabSize,
        dropout_rate= dropoutRate
    )
    # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        tf.optimizers.Adam(0.0001), loss=loss
    )
    # print(model.output_shape)
    return model


class TF_Gen(tf.Module):
    def __init__(self, model, TOK_MODEL, maxlen = 30, name="TextGen"):
        super().__init__()
        self.model = model
        self.inputP = tf_text.SentencepieceTokenizer(open(TOK_MODEL, "rb").read(), add_bos=True, add_eos=False)
        self.maxlen = maxlen

    def gen(self, inputs, maxlen):
        ks = 100
        insi = self.inputP.tokenize(inputs)
        imp = tf.shape(insi.to_tensor())[1]
        for i in range(imp, maxlen):
            aki = insi.to_tensor(shape=[None, maxlen])
            model_output = self.model(aki)[0, i-1, :]
            
            logits, indices = tf.math.top_k(model_output, k=ks, sorted=True)
            indices = tf.cast(indices, tf.int32)

            preds = tf.nn.softmax(tf.expand_dims(logits, 0))
            # print(preds)
            selected_index = tf.random.categorical(tf.math.log(preds), num_samples=1)
            selected_index = tf.squeeze(selected_index, axis=-1)

            sampled_index = tf.gather(indices, selected_index)
            print(sampled_index)
            insi = tf.concat([insi, tf.expand_dims(sampled_index, 0)], -1)
        
        return self.inputP.detokenize(insi.to_tensor())[0].numpy()
        # return insi.to_tensor()
    
model = Model(N=N)
model.build(input_shape=(None, maxlen,))
latest_checkpoint = tf.train.latest_checkpoint(modelDir)
model.load_weights(latest_checkpoint)
gen = TF_Gen(model, TOK_MODEL, maxlen= 10)

@tf.function
def predict_fn(prompt, maxlen):
    prompt = tf.expand_dims(prompt, 0)
    output = gen.gen(prompt, maxlen)
    return output

prompt = "오늘 더 나은 내일"
output = gen.gen(np.array([prompt]), 30)
print(bytes(output).decode("UTF-8"))

concrete_func = predict_fn.get_concrete_function(tf.TensorSpec([], tf.string), 100)

# gen._jit_compile = False
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [concrete_func],
    gen
)


converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS, # enable TF ops
    # tf_text.tflite_registrar.SELECT_TFTEXT_OPS,
]
converter.allow_custom_ops = True
converter.target_spec._experimental_custom_op_registerers = [
    "UnsortedSegmentJoin",
    "UpperBound",
    "SentencepieceTokenizeOp"
] 
converter._experimental_guarantee_all_funcs_one_use = True

print("converting")
generate_tflite = converter.convert()
print("converted")
run_inference("안녕", generate_tflite)

with open('NARE_GPT.tflite', 'wb') as f:
    f.write(generate_tflite)
    print("model save")
# generate_tflite = tf.lite.Interpreter(model_path="./model.tflite")
# generate_tflite.allocate_tensors()
# with open('modelOps2.txt', 'w') as f:
#     for i in generate_tflite._get_ops_details():
#         f.write(str(i))
#         f.write("\n")
#     print("model save")
sys.exit()
