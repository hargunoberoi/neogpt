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
- create shell script to run tokenizer and model ✅


[2025-05-27 18:02:46]

I spent the last couple of days trying to figure out how wandb works.

Now I want to wandbfy my code so that I can

1. Log metrics (not so interesting)
2. Save model weights (very interesting)
3. Set up continuous free training workflow (Cool!)

1 and 2 are must, which I want to tackle right now. 

Basic idea is that I should be able to store my weights somewhere, my code somewhere and be able to run it anywhere. 

The reason I am so motivated to do this is that Surojit and I spent roughly $2-3k in AWS gpu instances, although the whole process was slightly frustrating and at the end of it, we got some stupid T4 gpus which are available for free on google colab. Colab is also better to spin up, but the limitation is obviously

1. Where do you store your data (still unknown)
2. Where do you store your model weights (wandb!)
3. Shutdown of instances (to be tackled later)

So far now, I am trying to get it at least working with saved model and some basic tracking.

[2025-05-27 19:20:25]

Calling it a day today. I have successfully saved the tokenizer model in the artifacts.

Now I need to think how can I operationalize it simply. i.e without thinking about restarting it.

This would involve basic tracking and saving the model weights, but it would require some thinking.

I leave it to tommorrow

- Pass API key to google colab
- Save metrics
- Save model weights
- Run inference by calling model weights in a fresh colab

[2025-05-28 14:01:47]

Alright! Here we are.

Our model brain is at wandb ;
Model training happens on colab ; 
Code resides on github

This is legen... (wait for it) dary! 

A few things that I observed;

- Fix generations to be a table for better viewing
- The colab training ended very quickly (inactivity?)
- How do I resume a crashed run?

What should I tackle first?

I tackled the colab inactivity problem. Apparently the solution I have is the only one so I hope that works.

I'll start a fresh run and see how it holds up.

I am a little bit zoned out at the moment. Again, the same stuff that happens - lost in code.

I need a colab run attempt, but I'm not sure if everything basically works.

[2025-05-28 16:43:24]

Fixed a small bug in the code wherein I was not using n_layers.

This is the risk of "vibe coding" in neural networks. There are so many settings that it will be very difficult
to train properly if something is amiss.

Need to test out how long can kaggle work.

[2025-05-28 19:59:20]

Kaggle test succeeded!

Mission KGB (Kaggle Github wandB)

The idea is to use Kaggle for runtime, Github for code and wandB for hosting models.

I need to spam Kaggle though, which means I need a lot of email ids (and also a way to take care of it).

[2025-05-28 21:21:33]

Finally. I can use zoho as a means to create several accounts and then login to kaggle using them.

That means I basically have access to all the gpus under the sun.

I just need to setup my training properly.

What I need to do next is: 

Download ddp related material because I think I am ready for that now.

I could also do the additional exercises like improving the transformer model code, and doing the additional exercises as suggested by Andrej. 

Overall, from resource point of view, I do think I am set, although the dataset could be a catch. But I am betting that Huggingface would solve that problem.

[2025-05-29 12:56:04]

Alright, doing the challenges by andrej, I think I am ready for that.

EX1: The n-dimensional tensor mastery challenge: Combine the Head and MultiHeadAttention into one class that processes all the heads in parallel, treating the heads as another batch dimension (answer is in nanoGPT).

This means combining the AttentionHead code with Multi-Attention Head.

[2025-05-29 13:42:54]

Done! Wohoo! This was not as hard as I imagined, and the code is much shorter too! Super pleased!

[2025-05-29 13:58:00]

Exercise 2 somewhat finally makes sense, but I still am not clear how to implement it. 

I am making a strategic decision to not focus on it right now because I want to tackle other parts, especially the third problem which is somewhat related to training GPT-2.

Focusing on that seems to be more useful at the moment, but for a later time, I need to get a little more comfortable with pytorch.

[2025-05-29 16:01:52]

Training works! Success. Mission KGB is on.

Next up, I need to figure out how to train with a larger dataset. Or should I tackle ddp first?

[2025-05-29 17:41:29]

I've decided I'd first do the data training, i.e go to the big dataset and then figure out ddp.

Reason why I think this is the right approach is that ddp is optional, but figuring out how to work with large datasets is imperative.

Diving in to figure out how to use huggingface datasets library to make this happen.

[2025-05-29 18:57:22]

Surprisingly, this is so clearly taken care of. 

The huggingface datasets library is perfectly capable of meeting all the needs.

I don't even need to "download" the data anywhere. I just pull it on the fly and train.

What remains now is DDP, and then I think I am ready for the Andrej video (or put it better, then I can squeeze the most out of what was provided there).

Today's progress was spectacular but obviously built on past work.

1. I redesigned the gpt, I can now see it in my mind's eye
2. I figured out how to make a **free** online training process work.
3. What remains now?

A lot. I think there are few important concepts that I need to get on top of: 

- pytorch
- oop with python
- python iterators
- distributed training
- Handling special tokens in tokenizer

When I come back tomorrow, I think I should tackle ddp and squeeze Andrej video before I take up anything else. Also handling special tokens - even though I have incorporated them in hartokenizer, I don't understand it very well, and more depressingly I don't know how to incorporate it in training.

Anyway, time to take a break.

[2025-05-30 10:29:21]

Back. 

Let's tackle ddp.

[2025-05-30 16:02:02]

DDP is done. I understand it.

I'm moving to the video by Andrej, I think I should be able to squeeze it well.

[2025-05-30 17:59:04]

So my run abruptly ended, however it shows "finished". Not sure what happened there, but DDP gave the model a boost for sure.

But I have discovered a big weakness of this model.

I have trained the tokenizer on the stupid shakespeare dataset and then I am expecting it "figure out" how to do other stuff. The tokenizer is not only very limited, but also very poor. 

I'm thinking of leaving a long run tonight and that would mean pushing to complete the Andrej video tonight. This allows me all weekend to train (Basically two full days before I return on Monday).

I think this is a stellar idea (if I can pull it off that is)

First, let's rearrange the model so as to be able to load the weights from huggingface.

[2025-06-02 14:34:17]

The weight tieing is a stellar idea. Basically you save 30% in model size just by correctly identifying that embeddings and model outputs are basically the same idea, i.e taking tokens and getting rich representations from them.

I get it.

[2025-06-03 16:10:17]

Alright, I've done the "spectating" of the video, now it is time to implement stuff.

I might have to take a few steps back and forth. First, I think it will be a good idea to start crushing shakespeare again.

Things to do: 

- Get to old dataset (shakespeare) ✅
- replace harbpe with tiktoken for simplicity ✅
- apply speed boosting optimization ✅
- no wandb logging for now ✅
- no saving weights for now ✅
- add logging and weight saving logic within the model itself? (decide)

[2025-06-03 18:04:32]

New stuff: 

- Add learning rate scheduling ✅
- Add weight decay (and fused kernels for optimizers) ✅

[2025-06-03 18:15:50]

New stuff: 

- Add gradient accumulation ✅
- Add ddp

I think this is going to be the most important part. Grad accumulation should happen quickly, but ddp will be a challenge simply because I want to do ddp differently (based on what I learned)

But good news is both the steps can be easily tested with my kaggle integration

[2025-06-03 18:26:56]

Done with gradient accumulation, it was honestly not the most difficult step. 

That comes later 😃

[2025-06-03 18:30:26]

Which is now. 

Time for ddp. "Use all of your talent, I know you have a lot of it!"

[2025-06-03 19:30:20]

Looks like I have ddp done. I am expecting some bugs, but I'll take it out for a spin.

[2025-06-03 19:49:31]

DDP done again! My previous work coming real handy.

Now, it's time for to get the actual data, create a validation set, get an evaluation metric (part 1)

Then, saving model weights, and tracking data.

Pretty impressed by what I have so far. Wohoo!

[2025-06-04 11:31:14]

Alright. It's time to get data.

[2025-06-04 12:47:43]

Fineweb script understood and rewritten. I think it should work. I plan to run this on colab for faster speeds. But then again, stupid colab stops!

[2025-06-04 14:19:32]

I have all the files. Now it's time to work some wandb magic.

[2025-06-04 16:08:00]

Wandb magic has been pending because I am hitting limits with the storage on kaggle. However, no need to be afraid. I saw that kaggle limits only on one type of disk, however it's /mnt has a tb of storage it allows access to. Kaggle is "simply lovely"

Until then, I also sorted the logic of loading shards in an optimal way. That means, I can now test training the actual fineweb data. This is cool.

What is left: 

- Validation set (achievable)
- Hellaswag code
- Logging
- Model saving
- Dry run

[2025-06-04 16:56:15]

To my great disappointment, kaggle kernel died as well. What the hell. It used to be one of the more reliable kernels.

Anyway.

I think it's a good idea to set up validation code and hellaswag with it while I'm at it

[2025-06-05 12:53:24]

Alright, after several failures, I've managed to download the data on my laptop itself. Wohoo! 

AWS has not worked disappointingly. I don't think so I will get spot instances. I have to think about getting the dedicated instances.

I need to think what all is left right now.

[2025-06-05 16:01:00]

So both methods worked!

I have a tar file and I have it on kaggle. It's time for the test run now. I don't know if everything is set

[2025-06-05 18:43:54]

So the last three hours were spent in debugging some issue that comes with this massive dataset.

For now I am disabling sampling and will take care of it later.

Let's test at least if the code is working

[2025-06-05 19:19:47]

Training works on a single gpu, but ddp still failed. 

So I am creating an iterable dataset instead and test how it goes. 

If all else fails, I have the dataloaderlite (which I want to avoid because of the issues it created with the training)

[2025-06-06 12:00:32]

Now we have 8 A100s.

Time to train like a pro!

[2025-06-06 14:00:48]

Alright. It runs on aws. Let's do this! 

I think first run should be on 10 billion tokens.

But I should do it such that I could restart training from where it stopped.

This way, the initial training run will not be wasted. 

How do I do this?

Things to do 

- Peformance improvement: autocast ✅
- wandb model saving ✅
- try kaggle dataset download directly
- starting stopping instance from terminal command?
- logging of training loss, validation loss ✅
- cleanup

[2025-06-06 15:22:41]

Alright! I need to test out whether the stuff that I have done or not works. 

I am a little bit lost right now because there's a ton of stuff that needs to be done and simultaneously.

The thing I want to test is kaggle dataset direct download. this poses another challenge, how to share secrets?

The idea is to run an instance easily from creation (assuming it stops) I don't want to waste too much time restarting the run.

That poses a challenge. But for now, at least this much should run which I should test on kaggle gpus.

The "restart from previous state" is still questionable. I don't know what all I have to restore.

So

- test current use ✅
- restart run from old state
- kaggle dataset use ❌ (not better)

[2025-06-06 17:13:32]

I have decided against using kaggle because what the hell it uses a zip file. My method is better.

Only one thing left to do: restart from state

Alright. Looks fine. What is left now is to modify init to download the model.

[2025-06-06 18:06:41]

Alright. I think all is set; It is time to check this on kaggle first and then at night we start the mega training run.


[2025-06-06 20:21:12]

There are a bunch of issues here.

First, the logging bit is not working well despite setting the right level. I need to work on that later.

Also, torch.compile, annoyingly is not working right now.

I need the init.sh script to close the instance once it starts.

Restart is not working so well, so I don't know what to do.

[2025-06-07 07:32:02]

Alright. It's morning. The script did not run very well. The model "learns" something but it stil isn't making coherent sentences. 

I also suspect lot of issues at the code level so I am going to attempt to run andrej's code as is as the gold standard for performance.

I also need to delete the old model weights because it is consuming too much space.

I don't know where all the wandb code is because I remember being able to delete through wandb data quite easily (runs and artififacts) but for the next experiment, I don't think I'd need any of it.

[2025-06-19 13:52:56]

One new thing that I did but did not put down my thoughts are as follows: 

I tried the Andrej code and it worked (obviously)

But contrary to my belief, the issue was not with how I was loading the data. (I thought that I didn't get enough speed because DataLoaderLite was much faster), but in reality, all the missing speed was because of not using the right precision for the model weights.

One thing still remains. Am I confusing the model by give each gpu a different shard of data? 

I have super speed now. I need to test this out. But before that: 

- Remove wandb model weights tracking
- Figure out the logging library for good
- Test out with new code

I think I can do this. Obviously the end goal is that I want my own version of gpt2 that works, and then I can experiment with it to become the King of Punjab (or whatever)

I think I can do this.

[2025-06-19 15:21:20]

I've decided against logging too. It's something i can master later.

For now, let's remove the wandb weight saving and focus on running this and seeing if things are working or not.

The other test would be to do it with dataloaderlite and test it as well.