# Speech Commands Data Set v0.02 (Preprocessed for Oort)

This is a set of one-second .wav audio files, each containing a single spoken
English word. These words are from a small set of commands, and are spoken by a
variety of different speakers. The files are preprocessed and renamed for Kuiper's 
dataloader.

## Organization

The [dataset](https://fedscale.eecs.umich.edu/dataset/google_speech.tar.gz) is splited into training and testing set. Spoken words and speaker ids are encoded 
in each file name. If a speaker contributed multiple utterances of the same word, these are distinguished by the number at the end of the file name. For example, the file path `up_aff582a1_nohash_1.wav` indicates that the word spoken was "up", the speaker's id was "aff582a1", and this is the third utterance of that word by this speaker in the data set. Note that no details were kept of any of the participants age, gender, or location, and random ids were assigned
to each individual.

# References
 This dataset is covered in more detail at [https://arxiv.org/abs/1804.03209](https://arxiv.org/abs/1804.03209) and Its original location is at
[http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz).