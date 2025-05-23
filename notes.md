[2025-05-21 10:48:06]

Starting again with the transformer video.

At first I want to make sure I have everything in order.

That means, I know how to work with input text, tokenize, use a simple model and make a prediction.

"The whole shebang"

- [ ] Tokenize text
- [ ] Train bigram model
- [ ] Make inference

One problem I already forsee with your "bigram" approach is that constructing this matrix is close to impossible because I would have go through all the tokens.

(Tokens in the tokenizer or tokens in the training data?)

I think tokens in the training data. This sort of worked in the earlier example because our vocab size was limited (27 and 65) but now it is huge! 

Also, I suspect a problem if I can use the gpt tokenizer in the transformer too because my embedding matrix will blow up.
This sucks!

I have to scale back! :(

Or maybe not. Maybe I can use harbpe regex tokenizer and create a small token set with nice numbers.

I am keen to do this and I think it would even help me recapitulate what I learned from the previous lecture.

[2025-05-21 11:27:02]

Done! I have a tokenizer now, BPE style.

Next, I can take a crack at the bigram model (if I remember it!)

How would I do it?

[2025-05-21 15:18:31]

Done with bigram.

Both counts and neural network version.

The neural network version is the perfect hello world example in case of language modeling because it does everything - 
1. Data preprocessing
2. Tokenization
3. neural network architecture
4. inference

The only thing that changes for the transformer (or any other) version is the third step, i.e neural network architecture.

Let's tackle that, but before any complex stuff I need to figure out the attention block. That is the key piece of the puzzle. I have often struggled to visualize what happens after a batch of input with some context length is processed.

It starts with BxT, then becomes BxTxE ; for simplicity lets take B as 1; then it is TxE, this should become TxH so we would need a weight matrix ExH?

Yeah, we need three of these weight matrices for key, query and value.

The basic idea is to figure out how to update the TxE so that it has some shared information of everything in it's sequence.

Now how is that used to make a prediction?

I think we just stick a big linear layer that just gives us B,T,V from B,T,E; I can see the whole thing now - time to code.

[2025-05-21 17:44:27]

Alright. Transformer version with custom tokenization complete.

I do see the whole thing and it makes we swoon!

Input text -> Tokens -> embeddings -> contextual embeddings -> probabilities

The "activators" here are
- UTF-8 encoding
- token dict
- embedding table
- attention block
- linear layer

a trained model means we need everything in the "activators" section.

One thing is, I still don't "see" the training metrics, I am not saving the learned weights and I am not doing any hyper parameter tuning. Those are some improvements I need to make in an updated code to make it all work.

[2025-05-23 10:28:23]

I need to modify the codebase now to gear it up for better tracking and training.

First, I need wandb.

[2025-05-23 15:24:07]

I was wrong in thinking I needed wandb first.

I am a little paralyzed by how I have to set up the class methods but I am making a little progress.

Currently stuck at how I can get the metrics.

The old function by Andrej took a batch based on his batches function which I have replaced by dataloader.

Now, I need a good way to save and load model

[2025-05-23 16:44:33]

Alright. model loading done, model track stats done. Dataset, dataloader done. Now I just need a shell script that basically does things if models (tokenizer and actual trained language model don't exist). It should be pulling harbpe too because that is required to train tokenizer.


TooDoo

- make proper harbpe repository ✅
- create shell script to run tokenizer and model
