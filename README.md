# Dysarthric speech: Intelligibility improvement

The idea behind this project is to make speech recognition more better for the people with speech pathology. This makes the assisstive tools more responsive towards the speaker with speech disability.

## Algorithm
![Block_Diagram](https://user-images.githubusercontent.com/47143544/56799294-45cb8800-6836-11e9-9591-398fbedf8859.jpg)

we have used direct feature-based mapping technique to train all themodels. We use Automatic Speech Recognition (ASR) system at the end to measure Phoneme Accuracy for a specific speaker. Based on the phoneme accuracy, we evaluate the performance of the proposed conversion system.

## Automatic Speech Recognition (ASR)

Our ASR system was built using the KALDI toolkit to recognize the dysarthric speech. As dysarthric speech dataset, namely, Universal Access (UA) consists of single word sentences, we used phone-based acoustic modeling. Hence, at the backend  monophone-based Hidden Markov Model (HMM) is used as an acoustic model to train the system.
