# OCR assignment report
## Feature Extraction 
For Feature Extraction I used a pca algorithm which computes the pca matrix.
It is quite good because it dramatically reduces the data in size
Which is then added to the model.
I experimented with many principal components in order to figure out 
which ones were the best and arrived to the conclusion
that features 2 to 11 give the best scores.
I experimented by implementing feature selection but 
the score increase was either very small or nonexistent so I only used 
pca.

## Classifier 
I used a nearest-neighbour classifier as it proved most eficient.
It is somewhat computationally expensive but it makes so asumptions which makes it very reliable.
After reducing the features to 10 dimentions I use a median filter
to reduce the noise of the test pages that are to be classified.
I also did implement adding noise to the training data which
increased the accuracy quite substantially.
Originally it was picking random pages from the training data and adding noise to them.
This proved to be somewhat unreliable because of the randomness factor.
I changed it afterwards so that it would add noise to the first 2 pages of the set as
that proved to be the most efficient aproach after experimenting a bit.
Due to the way I add noise I had to set a seed in the function so that it would always
add it the same. This way I could ensure the same result on multiple runs


## Error Correction 
I attempted to create some form of error correction that searches for words
in a dictionary but encountered trouble when trying to separate words so I could not proceed.
It would have worked by adding a dictionary into the model. Separating the words on the pages.
Searching for the words in the dictionary until it finds something that's 1 letter off and replacing
that letter with the correct one.
This approach does however have it's downsides as it can make mistakes like replacing a letter that was
correct because of some words that are very similar.

## Performance
The percentage errors (to 1 decimal place) for the development data are
as follows:
- Page 1: 97.1%
- Page 2: 97.2%
- Page 3: 89.8%
- Page 4: 76.5%
- Page 5: 61.8%
- Page 6: 53.4%

## Other information
When adding too much noise to the testing pages or to too many of them 
the scores on the first couple of pages drops quite significantly.