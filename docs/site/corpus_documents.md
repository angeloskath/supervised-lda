<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
# Corpus

In **LDA++** *Corpus* wraps the actual corpus, which is simply a collection of
documents. The Corpus interface is very simple, with merely three methods.

```cpp
/**
 * A corpus is a collection of documents.
 */
class Corpus
{
    public:
        /** The number of documents in the corpus */
        virtual size_t size() const = 0;
        /** The ith document */
        virtual const std::shared_ptr<Document> at(size_t index) const = 0;
        /**
         * Shuffle the documents so that the ith document is any of the
         * documents with probability 1.0/size() .
         */
        virtual void shuffle() = 0;
};
```

**LDA++** contains three additional generic corpus types that could be easily
used for a multitude of applications.

1. EigenCorpus
2. ClassificationCorpus
3. EigenClassificationCorpus

## EigenCorpus

## ClassificationCorpus

*ClassificationCorpus* is useful for applications where the class relevant
information for every document in the corpus is important. Each document may
belong to one of the $C$ discrete classes and we want to compute the prior of a
specific class, let it be $y$, namely how many documents belong to this
specific class, divided by the total number of documents in the corpus.

```cpp
/**
 * A corpus that contains information about the classes of the documents in the
 * corpus (namely their priors).
 */
class ClassificationCorpus : public Corpus
{
    public:
        /**
         * @param  y A class
         * @return The count of the documents in class y divided by the count
         *         of all the documents
         */
        virtual float get_prior(int y) const = 0;
};
```

##
# Documents

