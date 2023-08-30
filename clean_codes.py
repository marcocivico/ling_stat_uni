#%% Importing packages
import numpy as np
import random
import pandas as pd
import fathon
from fathon import fathonUtils as fu
from matplotlib import pyplot as plt
from collections import Counter
from scipy.special import zeta
from scipy.optimize import bisect
from matplotlib import pyplot as plt
import nltk
import nltk.corpus
from collections import Counter
from nltk.tokenize import word_tokenize
#nltk.download('wordnet')
import regex as re
import powerlaw
from sklearn.linear_model import LinearRegression

#%% Load corpus - change encoding accordingly
author = "Jules_Verne"
corpus=[]
with open("Final_corpora/BramStoker_Dracula.txt", "r", encoding="ansi") as file:
    corpus = file.read()

#---MARKOV CHAIN MODELS---#

#%% Create transition table
def make_transition_table(corpus, ngram=5):
    dictionary = {}    
    for i in range(len(corpus) - ngram):
        X = corpus[i:i+ngram]
        Y = corpus[i+ngram]
        if dictionary.get(X) is None:
            dictionary[X] = {}
            dictionary[X][Y] = 1
        else:
            if dictionary[X].get(Y) is None:
                dictionary[X][Y] = 1
            else:
                dictionary[X][Y] +=1    
    return dictionary

#%% Convert transition table into probabilities
def freq_to_prob(dictionary):
    for kx in dictionary.keys():
        s = float(sum(dictionary[kx].values()))
        for ngram in dictionary[kx].keys():
            dictionary[kx][ngram] = dictionary[kx][ngram]/s            
    return dictionary

#%% Sample function
def sample_next(context, dictionary, ngram=20):
    context = context[-ngram:]        
    if dictionary.get(context) is None:
        return ' '
    possible_chars = list(dictionary[context].keys())
    possible_probabs = list(dictionary[context].values())
    return np.random.choice(possible_chars, p=possible_probabs)

#%% Generate samples
def generate_text(starting_sent, dictionary, ngram=20, max_len=100):
    sentence = starting_sent    
    context = sentence[-ngram:]    
    for i in range(max_len):
        next_pred = sample_next(context, dictionary, ngram)
        sentence += next_pred
        context = sentence[-ngram:]        
    return sentence

# %% Create ngram 4 samples for each ngram (2 through 10)
for i in range (4):
  for ngram in range(2,11):
      transition_table=make_transition_table(corpus, ngram=ngram)
      probabilities = freq_to_prob(transition_table)
      sample=generate_text(starting_sent=random.choice(list(probabilities.keys())), dictionary=probabilities, max_len=500000, ngram=ngram)
      with open(f'Final_corpora/Markov_samples/Verne_samples/{author}_{ngram}gram_sample_{i+1}.txt', 'w') as f:
          print(sample, file=f)

#---DETRENDED FLUCTUATION ANALYSIS---# fathon documentation: https://fathon.readthedocs.io/en/latest/index.html

#%% Prepare data: transform list of symbols into numerical sequence of their respective ranks
regular_expression = r'[^\w\s]'
corpus = re.sub(regular_expression, ' ', corpus)
corpus = re.sub("\s\s+" , " ", corpus)
def corr(s):
    return re.sub(r'\.(?! )', '. ', re.sub(r' +', ' ', s)) # add space after period

sequence = list(corpus)
char_freq = pd.DataFrame(Counter(sequence).items(),columns=["character","frequency"]).sort_values(by="frequency",ascending=False)
char_freq['rank'] = range(1, len(char_freq) + 1)
num_series=[]
for token in sequence:
    num=char_freq.loc[char_freq['character'] == f"{token}", 'rank'].iloc[0]
    num_series.append(num)

#%% Calculate zero-mean cumulative sum and initialize dfa object
cum_sum = fu.toAggregated(num_series)
pydfa = fathon.DFA(cum_sum)

#%% Compute fluctuation function and Hurst exponent
wins = fu.linRangeByStep(100, 10000, 10)
n, F = pydfa.computeFlucVec(wins, revSeg=True, polOrd=1)
H, H_intercept = pydfa.fitFlucVec()

#%% Perform DFA on shuffled sequence
shuffled_num_series = random.sample(num_series, len(num_series))
cum_sum_shuffled = fu.toAggregated(shuffled_num_series)
pydfa_shuffled = fathon.DFA(cum_sum_shuffled)
wins = fu.linRangeByStep(100, 10000, step=10)
n_shuffled, F_shuffled = pydfa_shuffled.computeFlucVec(wins, revSeg=True, polOrd=1)
H_shuffled, H_intercept_shuffled = pydfa_shuffled.fitFlucVec()

#%% Perform DFA on all generated samples
exponents=[]
Fs=[]
for ngram in range(2,11):
    for k in range(4):
        corpus_sample=[]
        with open(f"Final_corpora/Markov_samples/Verne_samples/{author}_{ngram}gram_sample_{k+1}.txt", "r", encoding="ANSI") as file:
            corpus_sample = file.read()
        #regular_expression = r'[^\w\s]'
        #corpus_sample = re.sub(regular_expression, ' ', corpus_sample)
        #corpus_sample = re.sub("\s\s+" , " ", corpus_sample)
        sequence_sample = list(corpus_sample)
        char_freq_sample = pd.DataFrame(Counter(sequence_sample).items(),columns=["character","frequency"]).sort_values(by="frequency",ascending=False)
        char_freq_sample['rank'] = range(1, len(char_freq_sample) + 1)
        num_series_sample=[]
        for token in sequence_sample:
            num=char_freq_sample.loc[char_freq_sample['character'] == f"{token}", 'rank'].iloc[0]
            num_series_sample.append(num)
        cum_sum_sample = fu.toAggregated(num_series_sample)
        pydfa_sample = fathon.DFA(cum_sum_sample)
        wins = fu.linRangeByStep(100, 10000, step=10)
        n_sample, F_sample = pydfa_sample.computeFlucVec(wins, revSeg=True, polOrd=1)
        H_sample, H_intercept_sample = pydfa_sample.fitFlucVec()
        exponents.append(H_sample)
        Fs.append(F_sample)
        print(f"The exponent of sample {ngram}-gram file {k+1} is:", H_sample)

# %% Perform DFA on one sample per n-gram and generate plot
plt.clf()
plt.xscale('log')
plt.yscale('log')
plt.title(f'DFA plot - {author}')
plt.xlabel('log(L)')
plt.ylabel('log(F(L))')
for ngram in range(2,11):
    for k in range(1):
        corpus_sample=[]
        with open(f"Final_corpora/Markov_samples/Verne_samples/{author}_{ngram}gram_sample_{k+1}.txt", "r", encoding="ANSI") as file:
            corpus_sample = file.read()
        sequence_sample = list(corpus_sample)
        char_freq_sample = pd.DataFrame(Counter(sequence_sample).items(),columns=["character","frequency"]).sort_values(by="frequency",ascending=False)
        char_freq_sample['rank'] = range(1, len(char_freq_sample) + 1)
        num_series_sample=[]
        for token in sequence_sample:
            num=char_freq_sample.loc[char_freq_sample['character'] == f"{token}", 'rank'].iloc[0]
            num_series_sample.append(num)
        cum_sum_sample = fu.toAggregated(num_series_sample)
        pydfa_sample = fathon.DFA(cum_sum_sample)
        wins = fu.linRangeByStep(100, 10000, step=10)
        n_sample, F_sample = pydfa_sample.computeFlucVec(wins, revSeg=True, polOrd=1)
        H_sample, H_intercept_sample = pydfa_sample.fitFlucVec()
        print(f"The exponent of sample {ngram}-gram file {k+1} is:", H_sample)
        plt.plot(wins, F_sample, linestyle='dashed', label=f"{ngram}-gram")
plt.plot(wins, F, label="Original", color="black", linewidth=2)
plt.plot(wins, F_shuffled, label="Shuffled", color="grey", linewidth=2)
plt.legend(loc="best")

#%% DFA Dracula + plots
corpus=[]
with open("Final_corpora/BramStoker_Dracula.txt", "r", encoding="ansi") as file:
    corpus = file.read()
regular_expression = r'[^\w\s]'
corpus = re.sub(regular_expression, ' ', corpus)

corpus = re.sub("\s\s+" , " ", corpus)

sequence = list(corpus)
char_freq = pd.DataFrame(Counter(sequence).items(),columns=["character","frequency"]).sort_values(by="frequency",ascending=False)
char_freq['rank'] = range(1, len(char_freq) + 1)
num_series=[]
for token in sequence:
    num=char_freq.loc[char_freq['character'] == f"{token}", 'rank'].iloc[0]
    num_series.append(num)

indices = [i for i, x in enumerate(num_series) if num_series[i:i+7] == [34, 26, 27, 40, 25, 47, 46,]]

cum_sum = fu.toAggregated(num_series)
pydfa = fathon.DFA(cum_sum)
wins = fu.linRangeByStep(100, 10000, 10)
n, F = pydfa.computeFlucVec(wins, revSeg=True, polOrd=1)
H, H_intercept = pydfa.fitFlucVec()
shuffled_num_series = random.sample(num_series, len(num_series))
cum_sum_shuffled = fu.toAggregated(shuffled_num_series)
pydfa_shuffled = fathon.DFA(cum_sum_shuffled)
wins = fu.linRangeByStep(100, 10000, step=10)
n_shuffled, F_shuffled = pydfa_shuffled.computeFlucVec(wins, revSeg=True, polOrd=1)
H_shuffled, H_intercept_shuffled = pydfa_shuffled.fitFlucVec()

plt.plot(cum_sum, label="original")
for xc in indices:
    plt.axvline(x=xc, linestyle="dashed", color="black", linewidth=0.5)
plt.plot(cum_sum_shuffled, label="shuffled")
plt.legend(loc="best")
plt.xlabel('r(t)')
plt.ylabel('Cumulative sum')
plt.locator_params(axis='x', nbins=5)
#plt.savefig("Dracula_DFA2.pdf", format="pdf", dpi=300)
plt.loglog(wins, F, label="original")
plt.loglog(wins, F_shuffled, label="shuffled")
plt.legend(loc="best")
plt.xlabel('Window size')
plt.ylabel('F(L)')
#plt.savefig("Dracula_DFA_F.pdf", format="pdf", dpi=300)

#---ZIPF'S EXPONENT---#

# Do A or B #
#%% A: Load Brown corpus (already tokenized)
words = [word.lower() for word in nltk.corpus.brown.words() if word.isalpha()]

#%% B: Load and tokenize author's corpus
corpus=[]
with open("Final_corpora/Transformers/Jules_Verne_transformer_topk50_topp85_nre3_sample_1.txt", "r", encoding="utf-8") as file:
    corpus = file.read()
regular_expression = r'[^\w\s]'
corpus = re.sub(regular_expression, ' ', corpus)

words = []
for word in corpus.split(" "):
    word_tokenize(word)
    if word.isalpha():
        words.append(word.lower())

#%% Create rank-frequency table
counter_of_words = Counter(words)
counter_of_counts = Counter(counter_of_words.values())
frequencies=np.asarray(list(counter_of_words.values()))
n=len(frequencies)

#%% Alternatives to discrete MLE
alpha_hat_cont = 1 + n / sum(np.log(frequencies)) # Continuous MLE
alpha_hat_discr = 1 + n / sum(np.log(frequencies/0.5)) # Approximation of discrete MLE

print(alpha_hat_cont, alpha_hat_discr)

# %% Calculate discrete MLE through bisection method (note that zeta'(x)/zeta(x) is the log derivative of zeta(x))
x=frequencies
#x = word_array
n=len(x)
xmin=1
def log_zeta(x, xmin=1):
    return np.log(zeta(x, xmin))

def log_deriv_zeta(x, xmin=1):
    h = 1e-5
    return (log_zeta(x+h, xmin) - log_zeta(x-h, xmin))/(2*h)

t = -sum(np.log(x/xmin))/len(x)
def objective(x, xmin=1):
    return log_deriv_zeta(x, xmin) - t

a, b = 1.01, 10
alpha_hat = bisect(objective, a, b, xtol=1e-6)

#Find standard error of discrete MLE
def zeta_prime(x, xmin=1):
    h = 1e-5
    return (zeta(x+h, xmin) - zeta(x-h, xmin))/(2*h)

def zeta_double_prime(x, xmin=1):
    h = 1e-5
    return (zeta(x+h, xmin) -2*zeta(x,xmin) + zeta(x-h, xmin))/h**2

def sigma(n, alpha_hat, xmin=1):
    z = zeta(alpha_hat, xmin)
    temp = zeta_double_prime(alpha_hat, xmin)/z
    temp -= (zeta_prime(alpha_hat, xmin)/z)**2
    return 1/np.sqrt(n*temp)

print(alpha_hat, "±", sigma(n, alpha_hat))

#%% Alternatively, using the "powerlaw" package
fit = powerlaw.Fit(frequencies, discrete=True, estimate_discrete=False, verbose=False)
alpha_hat_discr = fit.alpha
xmin = fit.xmin
zipf_exp = 1/(alpha_hat_discr-1)
sigma_z = (1/(alpha_hat_discr-1))**2*fit.sigma

print(f"The estimated ccdf exponent is {alpha_hat_discr} with an xmin of {xmin}.")
print(f"The associated Zipf exponent is {zipf_exp} ± {(1/(alpha_hat_discr-1))**2*fit.sigma}")

#%% Compare with exponent found fitting a linear regression to log-transformed data
reg_log = LinearRegression().fit(np.log(np.arange(1, len(frequencies)+1, 1)).reshape(-1,1),np.log(sorted(frequencies, reverse=True)))
coef = reg_log.coef_

#%% Plot function
sorted_counter_of_counts = sorted(counter_of_counts.items(), key=lambda pair: pair[1], reverse=True)
freq_of_word_counts = np.asarray(sorted_counter_of_counts)[:,1]
rf = [(r+1, f) for r, f in enumerate(freq_of_word_counts)]
rs, fs = zip(*rf)
plt.clf()
plt.xscale('log')
plt.yscale('log')
#plt.xlim(right=100)
plt.title('Zipf plot')
plt.xlabel('rank')
plt.ylabel('frequency')
plt.plot(rs, fs, 'r-')
plt.show()

#%% Zipf exponent for all samples
for ngram in range (2,11):
    for k in range(4):
        with open(f"Final_corpora/Markov_samples/Verne_samples/{author}_{ngram}gram_sample_{k+1}.txt", "r") as f:
            corpus = f.read()
        regular_expression = r'[^\w\s]'
        corpus = re.sub(regular_expression, ' ', corpus)
        words = []
        for word in corpus.split(" "):
            word_tokenize(word)
            if word.isalpha():
                words.append(word.lower())
        counter_of_words = Counter(words)
        counter_of_counts = Counter(counter_of_words.values())
        frequencies=np.asarray(list(counter_of_words.values()))
        x=frequencies
        n=len(frequencies)     
        fit = powerlaw.Fit(frequencies, discrete=True, estimate_discrete=False, verbose=False)
        alpha_hat_discr = fit.alpha
        xmin = fit.xmin
        se = fit.sigma
        zipf_exp = 1/(alpha_hat_discr-1)     
        print(f"The exponent of Zipf's law for {ngram}-gram sample {k+1} is: {zipf_exp} ± {(1/(alpha_hat_discr-1))**2*se} with an xmin of {xmin}.")

# %% Zipf plot for all samples and original
corpus=[]
with open("Final_corpora/JulesVerne_novels (reduced).txt", "r", encoding="ansi") as file:
    corpus = file.read()
regular_expression = r'[^\w\s]'
corpus = re.sub(regular_expression, ' ', corpus)
corpus = corpus[0:499999]

words = []
for word in corpus.split(" "):
    word_tokenize(word)
    if word.isalpha():
        words.append(word.lower())

counter_of_words = Counter(words)
counter_of_counts = Counter(counter_of_words.values())
frequencies=np.asarray(list(counter_of_words.values()))
n=len(frequencies)

fit = powerlaw.Fit(frequencies, discrete=True, estimate_discrete=False, verbose=False)
figure = plt.clf()
fit.plot_ccdf(ax=figure, color="black", linewidth=2, label="original")
for ngram in range (2,11):
    for k in range(2,3):
        with open(f"Final_corpora/Markov_samples/Verne_samples/{author}_{ngram}gram_sample_{k+1}.txt", "r") as f:
            corpus = f.read()
        regular_expression = r'[^\w\s]'
        corpus = re.sub(regular_expression, ' ', corpus)
        words = []
        for word in corpus.split(" "):
            word_tokenize(word)
            if word.isalpha():
                words.append(word.lower())
        counter_of_words = Counter(words)
        counter_of_counts = Counter(counter_of_words.values())
        frequencies=np.asarray(list(counter_of_words.values()))
        x=frequencies
        n=len(frequencies)  
        fit = powerlaw.Fit(frequencies, discrete=True, estimate_discrete=False, verbose=False)
        fit.plot_ccdf(ax=figure, label=f"{ngram}-gram")
plt.legend(loc="best")
#plt.title(f'Zipf plot (ccdf) - {author}')
plt.xlabel('Word frequency')
plt.ylabel('p(X≥x)')
plt.savefig(f"{author}_ZipfPlotFull.pdf", format = 'pdf', dpi=300)

#---ENTROPY---#

# %% Functions to calculate the estimator in Kontoyiannis et al. (1998)
def contains(small, big):
    try:        
        big.index(small)//len(big[0])
        return True
    except ValueError:
        return False

def entropy(string):
    n = len(string)
    sum_gamma = 0
    for i in range(1, n):
        sequence = string[:i]
        for j in range(i+1, n+1):
            sub_sequence = string[i:j]
            if contains(sub_sequence, sequence) != True:
                sum_gamma += len(sub_sequence)
                break

    entropy = 1 / (sum_gamma / n) * np.log(n)
    return entropy

# %% Calculate entropy on original and shuffled corpora
with open("Final_corpora/Transformers/Jane_Austen_transformer_topk50_topp85_nre3_sample_1.txt", "r", encoding="utf-8") as file:
    corpus = file.read()
print(f"The entropy rate of the original file is:", entropy(corpus[0:50000]))

shuffled_corpus = ''.join(random.sample(corpus,len(corpus)))
print(f"The entropy rate of the shuffled original corpus is:", entropy(shuffled_corpus[0:50000]))

# %% Calculate entropy on original and shuffled samples
for ngram in range (2,11):
    for k in range(4):
        with open(f"Final_corpora/Markov_samples/Verne_samples/{author}_{ngram}gram_sample_{k+1}.txt", "r") as f:
            data = f.read()
        print(f"Entropy rate of the {ngram}-gram file {k} is:", entropy(data[0:50000]))
# %%
for ngram in range (2,11):
    for k in range(4):
        with open(f"Final_corpora/Markov_samples/Verne_samples/{author}_{ngram}gram_sample_{k+1}.txt", "r") as f:
            data = f.read()
        shuffled_copy = ''.join(random.sample(data,len(data)))
        print(f"Entropy rate of the shuffled {ngram}-gram file {k} is:", entropy(shuffled_copy[0:50000]))

#---LONGEST COMMON SUBSEQUENCE---#
#%%
def lcs(string1, string2, len1, len2):
    LCS = [[0 for k in range(len2+1)] for l in range(len1+1)]
    result = 0
    for i in range(len1 + 1):
        for j in range(len2 + 1):
            if (i == 0 or j == 0):
                LCS[i][j] = 0
            elif (string1[i-1] == string2[j-1]):
                LCS[i][j] = LCS[i-1][j-1] + 1
                result = max(result, LCS[i][j])
            else:
                LCS[i][j] = 0
    return result

#%% Calculate LCS 
string_1 = []
with open("Final_corpora/JaneAusten_novels.txt", "r", encoding="ansi") as file:
    string_1 = file.read()

for ngram in range (2,11):
    for k in range(4):
        string_2 = []
        with open(f"Final_corpora/Markov_samples/Verne_samples/{author}_{ngram}gram_sample_{k+1}.txt", "r", encoding="ansi") as file:
            string_2 = file.read()
 
        print(f'The length of the longest common subsequence between the original and {ngram}-gram sample {k+1} is:',
            lcs(string_1[0:10000], string_1[10001:20001], 10000, 10000))
# %%
