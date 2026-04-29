#### Article

### Contrastive learning enables epitope overlap
### predictions for targeted antibody discovery

Graphical abstract

Highlights
##### • Contrastive fine-tuning encodes epitope relationships into
antibody LLM embeddings

##### • AbLang-PDB identifies overlapping-epitope antibodies
across 250 protein families

##### • Both structural and binding data successfully teach epitope
specificity

##### • AbLang-PDB achieved a 50% hit rate, identifying 8ANCoverlapping-epitope antibodies

Authors

Clinton M. Holt, Alexis K. Janke,
Parastoo Amlashi, Parker J. Jamieson,
Toma M. Marinov, Ivelin S. Georgiev

Correspondence
[ivelin.georgiev@vumc.org](mailto:ivelin.georgiev@vumc.org)

In brief

Antibodies are important for human
health, but their diversity makes
computational prediction of their binding
properties challenging. Through
contrastive fine-tuning of antibody
language models on millions of antibody
pairs, the authors enable accurate
prediction of epitope overlap even among
sequence-diverse antibodies, providing
powerful new tools for therapeutic
antibody discovery.

<||WXb23TXrUn3Rxz00yNNr89HV||>###### Article
### Contrastive learning enables epitope overlap
### predictions for targeted antibody discovery

Clinton M. Holt, ^1^ ^,^ ^2^ ^,^ ^3^ Alexis K. Janke, ^1^ ^,^ ^4^ Parastoo Amlashi, ^1^ ^,^ ^4^ Parker J. Jamieson, ^1^ ^,^ ^4^ Toma M. Marinov, ^1^ ^,^ ^3^ ^,^ ^4^
and Ivelin S. Georgiev ^1^ ^,^ ^2^ ^,^ ^3^ ^,^ ^4^ ^,^ ^5^ ^,^ ^6^ ^,^ ^7^ ^,^ ^8^ ^,^ ^9^ ^,^ ^10^ ^,^ ^11^ ^,^ *
1 Vanderbilt Center for Antibody Therapeutics, Vanderbilt University Medical Center, Nashville, TN 37232, USA
2 Program in Chemical and Physical Biology, Vanderbilt University Medical Center, Nashville, TN 37232, USA
3 Center for Computational Microbiology and Immunology, Vanderbilt University Medical Center, Nashville, TN 37232, USA
4 Department of Pathology, Microbiology, and Immunology, Vanderbilt University Medical Center, Nashville, TN 37232, USA
5 Department of Computer Science, Vanderbilt University, Nashville, TN 37232, USA
6 Department of Biomedical Informatics, Vanderbilt University, Nashville, TN 37232, USA
7 Department of Chemical and Biomolecular Engineering, Vanderbilt University, Nashville, TN 37232, USA
8 Department of Biochemistry, Vanderbilt University, Nashville, TN 37232, USA
9 Vanderbilt Institute for Infection, Immunology, and Inflammation, Vanderbilt University Medical Center, Nashville, TN 37232, USA
10 Center for Structural Biology, Vanderbilt University, Nashville, TN 37232, USA
11 Lead contact
*Correspondence: [ivelin.georgiev@vumc.org](mailto:ivelin.georgiev@vumc.org)
[https://doi.org/10.1016/j.patter.2025.101419](https://doi.org/10.1016/j.patter.2025.101419)

SUMMARY

Computational epitope prediction remains an unmet need for therapeutic antibody development. We present
three complementary approaches for predicting epitope relationships from antibody sequences. First, by
analyzing approximately 18 million antibody pairs targeting around 250 protein families, we establish that
over 70% of heavy-chain complementarity-determining region 3 (CDRH3) sequence identity among antibodies
sharing both V genes reliably predicts overlapping epitopes. Second, we develop a supervised contrastive fine-
tuning framework for antibody large language models that enriches embeddings with epitope information.
Applied toSARS-CoV-2 receptor-binding-domain antibodies, this approach achieves97%total accuracy in pre-
dicting high levels of structural overlap. Third, we create AbLang-PDB, a generalized model achieving 5-fold
improvement in average precision over sequence-based methods and correlating strongly with epitope overlap
( ***ρ*** = 0.81). Experimental validation with HIV-1 antibody 8ANC195 shows that 70% of selected candidates demon-
strate HIV-1 specificity and 50% compete for binding. These models provide powerful tools for epitope-targeted
antibody discovery while demonstrating contrastive learning’s efficacy for encoding epitope information.

INTRODUCTION

THE BIGGER PICTURE Developing vaccines and antibody drugs requires identifying where antibodies bind
to target proteins. Current computational methods face a fundamental trade-off: simple sequence compar-
isons are reliable when applicable but miss most promising candidates, while complex structural approaches
have broader applicability but require significant computational resources and often produce inaccurate pre-
dictions. This bottleneck significantly slows therapeutic development, particularly for challenging targets
such as rapidly mutating viruses. We developed a machine learning approach that teaches antibody lan-
guage models to recognize when different antibodies will bind to overlapping antigen sites, even when the
antibody sequences are significantly different. Our approach offers a practical solution for rapidly screening
large antibody databases, potentially accelerating the discovery pipeline by identifying the most promising
candidates before expensive laboratory validation.

![a sign that says "don't eat the wrong way"](outputs/1-s2.0-S2666389925002673-main_image_2_8.png)

<||WXb23TXrUn3Rxz00yNNr89HV||>into immune responses to vaccines and pathogens. Despite their
clinical success, developing therapeutic antibodies remains
resource intensive, with epitope characterization—identifying
the specific region on an antigen where an antibody binds—
posing a significant bottleneck. ^2^ For example, in the develop-
ment of broadly neutralizing antibodies (bNAbs) against HIV-1,
epitope mapping is critical to ensuring efficacy across diverse
viral strains. ^3^

Epitope characterization typically proceeds through three
complementary approaches: (1) structural mapping to define
physical contact points between antibody and antigen, (2)
functional mapping to identify binding-critical residues through
mutation, and (3) competition binding experiments to group an-
tibodies that interfere with each other’s binding. Each approach
helps guide therapeutic development, whether identifying sites
of vulnerability on pathogens or developing complementary anti-
body combinations. ^4–6^

Understanding the similarities and differences (or the level of
overlap) between the epitopes of different antibody candidates
provides critical information that can be utilized when developing
antibody therapeutics. For example, in pandemic response ef-
forts against a newly emerging virus, the selection of two or
more non-competing antibodies that synergize to form a more
effective drug than either individual antibody can be critical for
counteracting potential virus escape. In other cases, identifying
multiple antibodies against the same functionally important
epitope can provide a larger set of candidates for further evalu-
ation, down-selection, and development.
While experimental approaches for antibody epitope charac-
terization are undoubtedly effective, computational approaches
can present an efficient and cost-effective alternative. Generally,
computational approaches can interrogate the relationship be-
tween antibody sequence features and epitope similarity in order
to predict the level of epitope overlap between antibody candi-
dates. These approaches range from direct comparisons of the
full amino acid sequence or just the complementarity-deter-
mining region 3 (CDR3) amino acid sequence within gene groups
to comparing predicted structures or predicted antigen-binding
residues. ^5^ ^,^ ^7–16^ While the direct sequence-based methods have
shown success in clustering functionally related antibodies, the
antibody sequence similarity thresholds utilized by these ap-
proaches have been rigorously validated for only a few antigens
and epitopes. ^5^ ^,^ ^8–10^ ^,^ ^17^ The indirect approaches allow for search-
ing a broader antibody sequence space, but levels of accuracy
are low and unable to detect overlapping-epitope antibodies us-
ing distinct structural mechanisms, such as targeting the same
site from different angles—an aspect that can significantly influ-
ence Fc effector functions and binding breadth. ^16^ ^,^ ^18–20^ This lim-
itation is particularly problematic when searching for therapeutic
candidates, where expanding the candidate pool beyond highly
similar structures could be necessary to overcome challenges
such as low yields or suboptimal binding properties. ^21^

ential, having been trained on millions of naturally occurring anti-
bodies through masked language modeling to capture both evolu-
tionary relationships and structural constraints within antibody
sequences. ^28^ ^,^ ^32^ However, like other current antibody language
models, these tools face a critical limitation: their pretrained em-
beddings naturally cluster by sequence identity and germline
gene usage, ^29^ ^,^ ^31^ making them more adept at finding similar se-
quences than functionally similar antibodies with divergent se-
quences. This highlights a fundamental need for methods that
can specifically learn the complex sequence patterns underlying
epitope recognition beyond simple sequence similarity.
Recent advances in machine learning, particularly contrastive
learning approaches, offer promising solutions to these limita-
tions. Contrastive learning provides a framework for teaching
models to recognize when two examples should be considered
similar or different, even when observers see no clear patterns in
their features. A useful analogy is signature recognition: while
one’s signature may vary between years and with different
pens, contrastive learning enables machine learning models to
recognize the fundamental similarities between signatures and
distinguish authentic signatures from forgeries. By applying
this approach to antibody analysis, we can explicitly train models
to recognize structural or functional epitope similarity even when
sequence similarity is low. Using carefully curated training data
from structural databases and high-throughput epitope mapping
experiments, we demonstrate how this approach can enrich
antibody language model embeddings with epitope-specificity
information while maintaining their broad understanding of anti-
body sequence space.
In this work, we address three key challenges in antibody
epitope prediction. First, we establish reliable sequence-based
thresholds for identifying overlapping-epitope antibodies,
providing a simple yet powerful tool for repertoire analysis. Sec-
ond, wedevelop and validate amodel using thewell-characterized
severeacuterespiratorysyndromecoronavirus 2receptor-binding
domain (SARS-CoV-2 RBD), where extensive epitope mapping
data enable us to demonstrate how targeted training can over-
come the germline bias of current language models. Finally, we
present a generalized model capable of predicting epitope rela-
tionships across diverse protein families (Pfams), which we vali-
date through the successful identification of antibodies targeting
overlappingepitopeswiththeHIV-1bNAb8ANC195,atherapeutic
candidate that targets a unique epitope on the HIV-1 envelope
(Env) protein. These advances provide a comprehensive frame-
work for computational epitope analysis, offering new possibilities
for therapeutic antibody discovery and optimization.

RESULTS

<||WXb23TXrUn3Rxz00yNNr89HV||>tions. First, the most predictive rule applies only to the small sub-
set of antibody pairs sharing both V genes. Second, even within
this subset, the threshold fails to identify 82% of antibody pairs
that do bind overlapping epitopes, resulting in a high false nega-
tive rate. These limitations suggest that while sequence identity
can provide absolute confidence in some cases, more sophisti-
cated computational approaches may be needed for broader
applicability in therapeutic antibody discovery.
To address these limitations, we next explored the ability of
antibody large language models to learn the rules of epitope
specificity. We focused on two domains: learning discrete
epitope bins within one antigen and learning continuous epitope
information across diverse Pfams. These approaches, detailed
in the following sections, demonstrate how modern computa-
tional methods can overcome the constraints of simple
sequence-based rules.

![‘Same Pfam, Same Heavy } Same Light](outputs/1-s2.0-S2666389925002673-main_image_4_3.png)

**Figure 1.** **V-gene usage and CDRH3 sequence identity define reliable thresholds for predicting overlapping epitopes**
A comprehensive analysis of antibody sequence features predictive of epitope overlap within the Structural Antibody Database (SAbDab). Scatterplots show
complementarity-determining region 3 (CDR3) sequence identity relationships between antibody pairs ( *n* = 1,909 antibodies, ∼ 1 .8 million pairs) categorized by
epitope relationship (columns) and V-gene sharing status (rows). The columns represent overlapping epitopes (left), non-overlapping epitopes on the same
protein family (middle), and different protein families (right). Rows indicate V-gene sharing patterns: both heavy and light V genes shared (top), only heavy V gene
shared (second), only light V gene shared (third), or neither V gene shared (bottom). The *x* axis shows CDRH3 amino acid sequence identity, and the *y* axis shows
CDRL3 amino acid sequence identity. Data density is represented by hexagonal binning with color scaling from minimum (dark red) through yellow to maximum
density (dark blue). Dashed vertical lines indicate the 70% CDRH3 identity threshold. Numbers in the bottom corners indicate pair counts within the half
designated by the line.

<||WXb23TXrUn3Rxz00yNNr89HV||>available for SARS-CoV-2 RBD antibodies ^36–38^ to develop and
validate a contrastive learning framework that could encode
epitope-specificity information directly into antibody sequence
embeddings ( Figures 1 and 2 ).
Our model, AbLang-RBD, builds upon the established AbLang-
Heavy and AbLang-Light chain language models through targeted
fine-tuning using a supervised contrastive learning frame-
work. ^28^ ^,^ ^39–41^ The architecture processes paired antibody se-
quences through a dual-stream transformer network—with separate transformer blocks per chain—followed by a six-layer
multi-layer perceptron that generates unified sequence embed-
dings ( Figure 1 A). We optimized these embeddings using super-
vised contrastive learning as described by Khosla et al., ^41^ which
simultaneously processes multiple positive examples within
each training batch, allowing the model to learn from groups of an-
tibodies targeting the same epitope rather than individual pairs
( Figure 1 B). By training on same-epitope antibodies that fall
outside our previously established V-gene and CDRH3 identity
thresholds, the model learns new antibody sequence patterns
indicative of shared epitope binding that are missed by our out-
lined V-gene and CDRH3 thresholds.
We trained the model using a previously characterized
set of 3,041 SARS-CoV-2 RBD antibodies binned into 12 epi-
topes based on deep mutational scanning (DMS) results. ^37^ ^,^ ^38^

![a series of photos showing a small model of a bird](outputs/1-s2.0-S2666389925002673-main_image_5_2.png)

![a green and white street sign with a flower](outputs/1-s2.0-S2666389925002673-main_image_5_3.png)

![a red and white striped kitty with a red heart](outputs/1-s2.0-S2666389925002673-main_image_5_4.png)

![a cartoon of a cat with a red bow](outputs/1-s2.0-S2666389925002673-main_image_5_5.png)

![a painting of a cat with a red ribbon on it](outputs/1-s2.0-S2666389925002673-main_image_5_8.png)

![Repe](outputs/1-s2.0-S2666389925002673-main_image_5_10.png)

![a sign that says "don't miss the holidays"](outputs/1-s2.0-S2666389925002673-main_image_5_11.png)

**A**

**B**

**Figure** 2 **.** **Contrastive** **learning** **framework**
**for** **encoding** **epitope-specificity** **informa-**
**tion in antibody sequence embeddings**
(A) Model architecture for AbLang-RBD and
AbLang-PDB and their embedding mechanism.
This framework uses frozen AbLang heavy and
light models with unfrozen low-rank adaptation
parameters, as well as an added 6-layer multi-
layer perceptron (MLP) network to create a unified
1,536D embedding for each antibody fed into the
model.
(B) Representation of the contrastive learning
approach for the ‘‘LLM encoder’’ framework de-
picted in (A). During training, embeddings of anti-
bodies binding overlapping epitopes (two blue
antibodies) are pulled closer together in repre-
sentation space, while embeddings of antibodies
binding non-overlapping epitopes (pink compared
to blue) are pushed apart.

The effectiveness of our epitope-spe-
cific encoding was further demonstrated
through dimensionality reduction anal-
ysis. t-Distributed stochastic neighbor embedding (t-SNE)
visualization ^42^ reveals that while the pretrained model’s embed-
dings show minimal epitope-based clustering (31.2% *k* -means
accuracy), ^43^ AbLang-RBD achieves near-perfect clustering of
training data (99.6%) and substantially improved clustering of
test data (54.6%) ( Figure 3 B). Notably, when test antibodies
were misclassified, 43% of errors still placed them within the cor-
rect RBD epitope class (out of 4 generally accepted classes), ^44^

<||WXb23TXrUn3Rxz00yNNr89HV||>K-means
Accuracy:
5 4.6%

K-means
Accuracy:
9 9.6%

K-means
Accuracy:
3 1.2%

K-means
Accuracy:
5 4.6%

**A**

**B**

**C**

***ρ*** **= -.08**
**r = -.1**

***ρ*** **= -.39**
**r = -.45**

***ρ*** **= -.29**
**r = -.32**

***ρ*** **= -.21**
**r = -.23**

**Max**

**Min**

**Pair**
**Density**

***ρ*** **= .1**
***p*** ***ρ*** **= 8e-65**
**r = .14**

**Max**

**Min**

**Pair**
**Density**

***ρ*** **= .25**
***p*** ***ρ*** **< 5e-300**
**r = .34**

***ρ*** **= .09**
***p*** ***ρ*** **= 2e-47**
**r = .18**

BSA Overlap BSA Overlap BSA Overlap

**D**
CDRH3 Identity, PDB Dataset

8 **2.7% Accuracy** 7 **4.4% Accuracy** 5 **6.0% Accuracy**

**Same Epitope**
**Different Epitopes**
**Decision Threshold**

AbLang AbLang-RBD Train vs Test AbLang-RBD Test

AbLang AbLang-RBD Train AbLang-RBD Test

AbLang AbLang-RBD Train AbLang-RBD Test AbLang-RBD Train vs Test

AbLang, PDB Dataset AbLang-RBD, PDB Dataset

Density

**Figure 3.** **AbLang-RBD learns to predict epitope relationships from binned deep mutational scanning data**
(A) Distribution of cosine similarities between antibody pairs binding the same (blue) or different (red) epitopes. The pretrained model’s performance is shown on
the left, while the fine-tuned model’s performance is shown for either the comparisons of train-to-test antibodies (middle) or comparisons of test versus test
antibodies (right). Optimal decision thresholds (dashed lines) were determined using validation data.
(B) t-SNE visualization of antibody embeddings colored by epitope class. These are split to show pretrained embeddings (left) or fine-tuned embeddings on train
(middle) and test (right) antibodies.
(C) Model performance assessed against continuous deep mutational scanning (DMS) data. Scatterplots show the relationship between antibody pair cosine
similarities ( *y* axis) and distance between weighted average spatial coordinates derived from DMS escape maps ( *x* axis). Hexagonal bins are colored by pair
density from minimum (dark red) to maximum (dark blue). Spearman’s ( *ρ* ) and Pearson’s (r) correlation coefficients are shown.
(D) Validation using structural data from the Protein Data Bank (PDB). Scatterplots compare CDRH3 sequence identity (left), pretrained AbLang (middle), and
AbLang-RBD (right) against buried surface area (BSA) overlap between antibody pairs. Spearman’s ( *ρ* ) and Pearson’s (r) correlation coefficients are shown; *p*
values ( *p* *ρ* ) shown correspond to the Spearman correlation. Threshold lines are drawn at a cosine similarity of 0.85 and a BSA overlap of 750 A ^˚^ ^2^ .

| 82.7% Acc | uracy |
|-----------|-------|

| | ρ = -.08 ρ
r = -.1 r | = -.39 ρ = -.= -.45 r = -.32 |
|--|----------------------|---------------------------------|

<||WXb23TXrUn3Rxz00yNNr89HV||>sequence-based methods. This capability, validated against both
DMS and structural data, provides a valuable new tool for thera-
peutic antibody discovery, particularly in cases where traditional
sequence similarity metrics fail to identify functionally related an-
tibodies. Most importantly, this framework establishes a founda-
tion for developing even more sophisticated models that can cap-
ture the continuous nature of epitope relationships across diverse
antigen families.

![a series of photos showing a person's hand](outputs/1-s2.0-S2666389925002673-main_image_7_5.png)

**A**

**B** **C**

**D** **E**

**Figure** 4 **.** **AbLang-PDB** **enables** **accurate**
**prediction** **of** **epitope** **relationships** **across**
**diverse protein families**
Evaluation of AbLang-PDB’s performance on the
Structural Antibody Database(SAbDab), comparing
antibodies in the training dataset to held-out anti-
bodies.
(A) Distribution of cosine similarities between anti-
body pairs categorized as overlapping epitopes
(blue), non-overlapping epitopes on the same pro-
tein family (yellow), or different protein families (red).
Both thepretrained (left)andfine-tuned(right)model
results are shown, as well as optimal classification
thresholds shown as dashed lines.
(B) Relationship between model-predicted cosine
similarities and ground-truth labels. Hexagonal bins
colored by pair density (white to dark blue). Black
bars indicate mean ± 95% confidence intervals.
Spearman correlations ( *ρ* ) and maximum possible
correlations ( *ρ* max ) are shown.
(C) A zoomed-in view of high-confidence pre-
dictions for overlapping-epitope pairs (cosine simi-
larity and label ≥ 0 .5). The Spearman corelation ( *ρ* )
and corresponding *p* value ( *p* ) are shown.
(D and E) Receiver operating characteristic curves
(D) and precision-recall curves (E) for several clas-
sification methods, both for predicting overlapping
epitopes (left) and for binding the same protein
family (right).

<||WXb23TXrUn3Rxz00yNNr89HV||>these categories. More importantly, this training paradigm
directly supports our intended application of mining large
sequence databases for antibodies with overlapping epitopes
relative to known therapeutic references.
These results demonstrate that our continuous learning
approach successfully captures epitope relationships across
diverse Pfams while maintaining high precision for overlap-
ping-epitope predictions. The model’s ability to provide reliable
confidence scores through cosine similarities makes it particu-
larly valuable for therapeutic antibody discovery, where false
positives can be costly.

Table 1. Benchmarked model performance

Pairs used Model

SARS-CoV-2 RBD DMS
dataset SAbDab

Same epitope Same Pfam Overlapping epitopes

AUROC Avg Prec F1 AUROC Avg Prec F1 AUROC Avg Prec FTrain versus test **AbLang-RBD** 0 **.84** 0 **.64** 0 **.59** 0 .51 0 .15 0 .19 0 .1. 16 ^a^
2. **AbLang-PDB** 0 .54 0 .15 0 .20 0 **.79** 0 **.51** 0 **.50** 0 **.81** 0 **.54** 0 **.56**

AbLangPre

1. 57 ^a^
2. 3. 21 ^a^
4. 59 0 .15 0 .21 0 .63 0 .08 0 .AbLang-Heavy 0 .56 0 .16 0 .21 0 .62 0 .18 0 .23 0 .65 0 .10 0 .AbLang2 0 .54 0 .13 0 .20 0 .54 0 .12 0 .19 0 .54 0 .05 0 .IgBERT 0 .54 0 .14 0 .20 0 .58 0 .15 0 .21 0 .60 0 .07 0 .BALM 0 .56 0 .16 0 .20 0 .59 0 .15 0 .22 0 .61 0 .07 0 .AntiBERTy 0 .57 0 .17 0 .1. 64 ^a^ 0 .19 ^a^ 0 .24 ^a^ 0 .66 ^a^
2. 3. 17 ^a^
ESM-2 0 .56 0 .18 0 .20 0 .55 0 .13 0 .19 0 .60 0 .08 0 .Parapred 0 .55 0 .15 0 .20 0 .54 0 .12 0 .19 0 .58 0 .06 0 .SEQID 0 .53 0 .14 0 .20 0 .53 0 .12 0 .18 0 .56 0 .07 0 .CDRH3ID 0 .1. 19 ^a^
2. 21 0 .54 0 .14 0 .18 0 .61 0 .09 0 .Test versus test AbLang-RBD 0 **.73** 0 **.39** 0 **.39** 0 .53 0 .21 0 .19 0 .61 0 .25 0 .AbLang-PDB 0 .53 0 .14 0 .20 0 **.68** 0 **.34** 0 **.33** 0 .68 0 **.33** 0 **.35**

AbLangPre

1. 57 ^a^
2. 16 0 .20 0 .63 0 .25 0 .20 0 .68 0 .21 0 .AbLang-Heavy 0 .56 0 .15 0 .20 0 .64 0 .1. 26 ^a^
2. 68 0 .19 0 .AbLang2 0 .53 0 .12 0 .20 0 .53 0 .15 0 .20 0 .56 0 .08 0 .IgBERT 0 .54 0 .13 0 .20 0 .60 0 .20 0 .23 0 .63 0 .14 0 .BALM 0 .55 0 .15 0 .20 0 .60 0 .22 0 .24 0 .63 0 .19 0 .AntiBERTy 0 .57 0 .1. 21 ^a^ 0 .66 ^a^ 0 .27 ^a^
2. 25 0 **.71**
3. 26 ^a^ 0 .30 ^a^
ESM-2 0 .56 0 .17 0 .20 0 .56 0 .19 0 .20 0 .60 0 .17 0 .Parapred 0 .55 0 .14 0 .20 0 .59 0 .20 0 .21 0 .63 0 .18 0 .SEQID 0 .1. 18 ^a^
2. 19 0 .58 0 .23 0 .3. 69 ^a^
4. 26 0 .CDRH3ID 0 .53 0 .13 0 .20 0 .54 0 .20 0 .20 0 .64 0 .23 0 .Performance metrics across all models for the SARS-CoV-2 RBD DMS dataset and the SAbDab dataset (subdivided into ‘‘same Pfam’’ and ‘‘overlap-
ping epitopes’’ tasks). Models are evaluated under two conditions: ‘‘train versus test’’ (comparing antibodies in the train dataset to the test dataset) and
‘‘test versus test’’ (comparing held-out antibodies against each other). The bold formatting indicates the best performance in a benchmark. Baseline
models include AbLang-Heavy (Olsen et al. ^28^ ); AbLang-Pre, which is a concatenation of AbLang-Heavy and AbLang-Light embeddings (Olsen et al. ^28^ );
AbLang2 (Olsen et al. ^29^ ); IgBERT (Kenlay et al. ^30^ ); BALM (Burbach and Briney ^31^ ); AntiBERTy (Ruffolo et al. ^27^ ); ESM-2 (Lin et al. ^25^ ); and Parapred (Liberis
et al. ^26^ ). In these benchmarks, Parapred was used by generating antibody-specific embeddings in a custom method. The final hidden state of the
model was extracted and averaged over all CDR residues in both chains to obtain the embedding. AUROC, area under the receiver operating curve;
Avg Prec, average precision; RBD, receptor-binding domain, DMS, deep mutational scanning; SEQID, total amino acid sequence identity over the
variable region; CDRH3ID, heavy chain complementarity-determining region 3 amino acid sequence identity.
a The second best performance in a benchmark.

| AbLang-RBD | 0.84 | 0.64 | 0.59 | 0.51 | 0.15 | 0.19 | 0.58 | 0.16a | 0.14 |
|--------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| AbLang-PDB | 0.54 | 0.15 | 0.20 | 0.79 | 0.51 | 0.50 | 0.81 | 0.54 | 0.56 |
| AbLangPre | 0.57a | 0.18 | 0.21a | 0.59 | 0.15 | 0.21 | 0.63 | 0.08 | 0.13 |
| AbLang-Heavy | 0.56 | 0.16 | 0.21 | 0.62 | 0.18 | 0.23 | 0.65 | 0.10 | 0.14 |
| AbLang2 | 0.54 | 0.13 | 0.20 | 0.54 | 0.12 | 0.19 | 0.54 | 0.05 | 0.08 |
| IgBERT | 0.54 | 0.14 | 0.20 | 0.58 | 0.15 | 0.21 | 0.60 | 0.07 | 0.12 |
| BALM | 0.56 | 0.16 | 0.20 | 0.59 | 0.15 | 0.22 | 0.61 | 0.07 | 0.12 |
| AntiBERTy | 0.57 | 0.17 | 0.21 | 0.64a | 0.19a | 0.24a | 0.66a | 0.12 | 0.17a |
| ESM-2 | 0.56 | 0.18 | 0.20 | 0.55 | 0.13 | 0.19 | 0.60 | 0.08 | 0.10 |
| Parapred | 0.55 | 0.15 | 0.20 | 0.54 | 0.12 | 0.19 | 0.58 | 0.06 | 0.10 |
| SEQID | 0.53 | 0.14 | 0.20 | 0.53 | 0.12 | 0.18 | 0.56 | 0.07 | 0.08 |
| CDRH3ID | 0.56 | 0.19a | 0.21 | 0.54 | 0.14 | 0.18 | 0.61 | 0.09 | 0.12 |
| AbLang-RBD | 0.73 | 0.39 | 0.39 | 0.53 | 0.21 | 0.19 | 0.61 | 0.25 | 0.22 |
| AbLang-PDB | 0.53 | 0.14 | 0.20 | 0.68 | 0.34 | 0.33 | 0.68 | 0.33 | 0.35 |
| AbLangPre | 0.57a | 0.16 | 0.20 | 0.63 | 0.25 | 0.20 | 0.68 | 0.21 | 0.19 |
| AbLang-Heavy | 0.56 | 0.15 | 0.20 | 0.64 | 0.23 | 0.26a | 0.68 | 0.19 | 0.23 |
| AbLang2 | 0.53 | 0.12 | 0.20 | 0.53 | 0.15 | 0.20 | 0.56 | 0.08 | 0.12 |
| IgBERT | 0.54 | 0.13 | 0.20 | 0.60 | 0.20 | 0.23 | 0.63 | 0.14 | 0.18 |
| BALM | 0.55 | 0.15 | 0.20 | 0.60 | 0.22 | 0.24 | 0.63 | 0.19 | 0.19 |
| AntiBERTy | 0.57 | 0.16 | 0.21a | 0.66a | 0.27a | 0.25 | 0.71 | 0.26a | 0.30a |
| ESM-2 | 0.56 | 0.17 | 0.20 | 0.56 | 0.19 | 0.20 | 0.60 | 0.17 | 0.20 |
| Parapred | 0.55 | 0.14 | 0.20 | 0.59 | 0.20 | 0.21 | 0.63 | 0.18 | 0.15 |
| SEQID | 0.55 | 0.18a | 0.19 | 0.58 | 0.23 | 0.20 | 0.69a | 0.26 | 0.13 |
| CDRH3ID | 0.53 | 0.13 | 0.20 | 0.54 | 0.20 | 0.20 | 0.64 | 0.23 | 0.17 |

<||WXb23TXrUn3Rxz00yNNr89HV||>3 fusion (HPIV3 F) protein as a negative control ( Figures 5 C, S3 A,
and S3B ^53^ ). Seven of the ten selected antibodies (70%) demon-
strated HIV-1 Env specificity, with six (60%) showing cross-clade
binding. Out of our unbiased selection, two had previously been
characterized—2723-3055 and 3602-870—both of which had
been shown to potently neutralize a broad panel of tier 2 HIV-1 vi-
ruses (12/12 and 11/14 viruses tested). ^53^

![c ELISA Binding 27232068 (1) 27236265](outputs/1-s2.0-S2666389925002673-main_image_9_3.png)

![a building with a clock on it](outputs/1-s2.0-S2666389925002673-main_image_9_4.png)

![a vase filled with flowers and a picture of a person](outputs/1-s2.0-S2666389925002673-main_image_9_5.png)

**A**

**B** **C**

**D**

**E** **F**

**Figure** 5 **.** **AbLang-PDB** **successfully** **iden-**
**tifies HIV antibodies that compete for bind-**
**ing with 8ANC195**
Experimental validation of AbLang-PDB pre-
dictions using HIV broadly neutralizing antibody
8ANC195.
(A) Sequence characteristics of top candidate
antibodies selected by cosine similarity to
8ANC195. The table shows model predictions,
sequence identity metrics, gene usage, and CDR
information for each antibody. Reference anti-
bodies 8ANC195 and VRC01 are included for
comparison.
(B) Distribution of cosine similarities across the
complete LIBRA-seq dataset ( *n* = 7,056 anti-
bodies), with dashed line indicating recom-
mended threshold (0.5) for mining overlapping-
epitope candidates.
(C) ELISA binding profiles against HIV-1 envelope
SOSIP.664 constructs (BG505, CZA97, ZM106.9)
and HPIV3 F control protein. Binding strength is
indicated by the area under the curve (white to
blue). The 3X1 antibody was included as an
HPIV3-specific control.
(D) Structural representation of HIV-1 BG505 en-
velope showing competitor antibody epitopes:
8ANC195 (green, gp120-gp41 interface), VRC(pink, CD4-binding site), and PG9 (tan, V1-V2 re-
gion) from PDB: 5VJ6, 8VGW. The envelope sur-
face is shown with gp120 (light gray) and gp(black).
(E) Competition ELISA curves showing percent-
age of reduction in binding of biotinylated
8ANC195 (10 μ g/mL) to BG505 SOSIP.664 in the
presence of increasing concentrations of blocking
antibodies. Filled symbols indicate mAbs dis-
playing competition with 8ANC195.
(F) Competition matrix showing percentage of
reduction in binding at fixed concentrations
(blocking: 100 μ g/mL; detection: 10 μ g/mL
8ANC195 and 1 μ g/mL VRC01 and PG9). Values
range from no competition (white) to complete
competition (black).

| | | | | | | | | | |
|--|--|--|--|--|--|--|--|--|--|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

<||WXb23TXrUn3Rxz00yNNr89HV||>Env specificity—highlights AbLang-PDB’s potential to streamline
therapeutic antibody discovery by accurately identifying function-
ally relevant candidates that conventional sequence similarity
metrics can miss. This is particularly noteworthy given that the
model identified two previously validated bNAbs without any prior
knowledge of their functional properties. These results support
AbLang-PDB’s utility for therapeutic antibody discovery, espe-
cially in cases where conventional sequence similarity metrics
would fail to identify functionally related candidates.

DISCUSSION

The identification of antibodies targeting overlapping epitopes re-
mains a critical challenge in therapeutic antibody development.
Current approaches typically rely on experimental screening ^57^

mance, AbLang-PDB accurately estimates epitope overlap
through cosine similarity scores, with high-confidence predictions
(cosine similarity > 0 .5) for overlapping-epitope antibody pairs
showing strong correlation ( *ρ* = 0.811) with actual epitope overlap
( Figure 4 C).
Our model demonstrated therapeutic relevance through the
identification of HIV-specific antibodies sharing epitope overlap
with 8ANC195. Of our 10 computationally selected candidates,
70% showed HIV-1 specificity, 50% competed with 8ANCfor binding, and 20% were broadly neutralizing. The 2 bNAbs
had both been previously discovered, but of note, they were
the only two previously discovered bNAbs present among the
7,056 mAbs in the database. Feature attribution analysis using
integrated gradients of the five experimentally validated
8ANC195-competing antibodies revealed unexpected insights
into AbLang-PDB’s decision-making process ( Figure S4 ). ^58^

<||WXb23TXrUn3Rxz00yNNr89HV||>but faces two key constraints: it is not clear that it will gener-
alize to RBD-specific antibodies incapable of binding the in-
dex strain and its training approach using epitope bins from
DMS data has not been validated to show that this will work
using epitope bins from the more commonly available anti-
body-antibody competition binding data. AbLang-PDB’s
training paradigm is optimized for comparing novel antibodies
against reference antibodies from its training set rather than
directly comparing two previously unseen antibodies. Addi-
tionally, this training paradigm is likely influenced by the
specific training labels chosen by hand ( − 1, 0 .2, and 0 .5–

1. 0), which represent heuristic choices that could potentiallybe optimized through systematic hyperparameter search ap-
proaches. Furthermore, while our HIV-1 validation demon-
strates practical utility, this antigen is well represented in
our training data, though notably, our validation epitope
(8ANC195’s epitope) had minimal representation.
Despite these limitations, our work provides a comprehensive
framework for computational epitope analysis that will be of sig-
nificance for the field of therapeutic antibody discovery. The
combination of simple sequence rules and sophisticated ma-
chine learning models offers researchers a tiered approach to
identifying overlapping-epitope antibodies, from rapid initial
screening to detailed prediction of epitope relationships. Look-
ing forward, these methods can be further enhanced through
integration with emerging structural prediction tools and
expanded training datasets, potentially enabling even more ac-
curate prediction of antibody-antigen interactions.

METHODS

Data curation
We curated a comprehensive dataset from the SAbDab ^33^ ^,^ ^34^

(February 19, 2024, cutoff date) for training and validating the
AbLang-PDB model. Starting with 16,105 antibody-antigen
complexes, we applied the following filtering criteria: resolution
≤ 4 .5 A ^˚^ , human antibodies with both chains present, and ≥ amino acid differences between antibodies. This yielded 1,non-redundant complexes, of which 184 had no same-Pfam
pairs and 485 had no overlapping-epitope pairs.
Antigen classification utilized pfam_scan ^35^ ^,^ ^59^ software to
group antigens by domain architecture using hidden Markov
models. Multiple Pfam assignments were consolidated such
that any shared Pfam between antigens classified their respec-
tive antibodies as targeting the ‘‘same Pfam’’ and thus a machine
learning label of 0 .2 or greater. When no overlap was present,
these pairs were assigned a machine learning label of − 1 . For
quantifying epitope overlap, we employed two complementary
approaches.
*Approach 1: BSA*
Here, we calculated BSA per residue using the Shrake-Rupley ^60^

antibody or after synthetically removing the antibody from the
structure file, such that *BSA* *residue* *i* =
*ASA* *residue* *i* *;* *Ag* *only* −
*ASA* *residue* *i* *;* *Ag* *in* *complex* . Per-residue BSA overlap between two
antibody-antigen complexes was calculated as

*BSA* *OVERLAP* *residue* *i* = *min*
(
*BSA* *residue* *i* *;* *complex* 1 *;*
*BSA* *residue* *i* *;* *complex* )
*;*

where *residue_i* refers to equivalent residues in each of the complexes as defined by a pairwise sequence alignment of the
antigen sequences. Pairwise sequence alignments of the anti-
gen sequences were accomplished using the BLOSUM62 ma-
trix ^63^ and Needleman-Wunsch ^64^ algorithm. Antibody pairs with
total BSA overlap summed over all residues ≤ 20 A ^˚^ ^2^ were labeled
as non-overlapping ( Figure S1 A, label = 0.2).
*Approach 2: Distance and volume overlap*
In this approach, we defined epitopes as antigen heavy atoms
within 4 .5 A ^˚^ of antibody atoms, and these selections were
accomplished using PyMOL. ^65^ For all antibody-antigen com-
plexes sharing at least one Pfam label, the antigens were aligned
in PyMOL. Next, the volume overlap of epitope atoms was calcu-
lated using PyMOL’s overlap function, with pairs showing over-
lap of ≤ 5 A ^˚^ ^3^ labeled as non-overlapping (label = 0.2).
For overlapping epitopes, final labels were assigned on a
continuous scale from 0.5 to 1.0 using the formula

*label* = *min*
(
1 *;* 0 *:* 5 + ( *rBSA* *OVERLAP* + *rATOM* *OVERLAP* ) ^0^ *^:^* ^75^ ^)^ *;*

where *rBSA_OVERLAP* and *rATOM_OVERLAP* represent over-
lap relative to the smaller of the two self-overlap values seen
for each epitope pair. For partitioning antibodies between data-
sets, antibodies sharing both heavy and light V genes and
CDRH3 amino acid identity >65% were assigned to the same
clone group. These groups were then distributed across training
(80%), validation (10%), and test (10%) sets, ensuring no clone
group appeared split between multiple sets. Additionally, pairs
with >92.5% sequence identity in either chain were excluded
to maintain diversity.
For AbLang-RBD, we utilized published DMS data
comprising 3,195 antibodies from 2 papers, ^37^ ^,^ ^38^ of which only
the 3,093 that demonstrated binding to the SARS-CoV-2 index
strain were kept. These antibodies were clustered based on
heavy-chain V-gene usage and CDRH3 amino acid identity
>70%, with clusters distributed across training (80%), valida-
tion (10%), and test (10%) sets such that no antibodies in the
same cluster existed in the training and test sets. A separate
test set was curated from the PDB by selecting RBD-specific
antibodies from the Coronavirus Antibody Database (CoV-
AbDab) ^36^ that demonstrated index strain binding and were
unique from those in the DMS dataset. This left 237 antibodies
and 27,345 pairs.

<||WXb23TXrUn3Rxz00yNNr89HV||>The base architecture follows RoBERTa ^67^ with modifications for
antibody sequence processing: each chain is processed through
12 transformer blocks containing 12 attention heads, with a hid-
den dimension of 768 and an intermediate dimension of 3,072. A
learned positional embedding layer handles sequences up to a
length of 160.
For sequence processing, antibody amino acid sequences
were first tokenized using the transformers module. Heavy-
and light-chain sequences were processed independently
through their respective models to generate chain-specific em-
beddings. For each chain, the final hidden layer outputs (768D
vectors) from all non-masked positions were mean pooled.
Whereas for the pretrained AbLang model, we simply concate-
nated these chain embeddings, the architecture for AbLang-
RBD and AbLang-PDB introduces additional processing layers
to enable cross-chain information flow. Specifically, the concat-
enated 1,536D vector (768 dimensions per chain) is processed
through a 6-layer multi-layer perceptron with rectified linear
unit (ReLU) activation between layers, except for the final layer.
The normalized output of this network serves as the unified anti-
body embedding.
To enable efficient fine-tuning while preserving pretrained
weights, we employed QLORA (quantized low-rank adaptation)
with rank R = 16, alpha = 32, and dropout = 0 .3. ^39^ This dual-
stream architecture—with 12 transformer blocks per chain, fol-
lowed by the cross-chain mixing network—allows the model to
capture both chain-specific features and relationships between
heavy- and light-chain sequences.

AbLang-RBD training
The AbLang-RBD model was trained using a supervised
contrastive learning approach to differentiate antibody embed-
dings based on their epitope label. Specifically, we employed
the supervised contrastive loss function, as introduced by Kho-
sla et al. ^41^ This loss function is designed to pull embeddings with
the same label closer together in the embedding space while
pushing apart those with different labels.
Training was performed with a batch size of 256 antibodies.
Optimization was performed using the AdamW optimizer with a
learning rate of 1e − 5 . During training, we froze all pretrained
weights except for the QLORA adaptation parameters and the
six ‘‘mixing’’ layers that enable crosstalk between heavy- and
light-chain embeddings. The detailed mathematical formulation
of the loss function is outlined below.
*Contrastive loss mathematical formulation*
Notation:

(1) Batch size of antibodies: *B* = 2 56.
(2) Embeddings: *z* *i* ∈ *R* ^1,536^ *for i* = 1, … , *B* .
(3) Epitope labels: *y* *i* ∈ *Z for i* = 1, … , *B* .
(4) Temperature parameter: τ = 0.5.
(5) Set of positive pairs for anchor antibody *i* : P ( *i* ) = { *j* ⃒⃒ *y* *i* =
*y* *j* *;* *j* ∕ = *i* } .

*Step* *1:* *Similarity* *matrix.* Compute the similarity matrix *S* ∈
*R* *^B^* ^×^ *^B^* , where each element represents the scaled cosine similar-
ity between two antibody embeddings:

*Step 2: Numerical stability.* For numerical stability during expo-
nentiation, subtract the maximum value from each row of the
similarity matrix:

*S* *ij* = *S* *ij* − max

*k*
*S* *ik* *:*

*Step 3: Denominator calculation.* For each antibody *i* , calculate
the denominator *D* *i* by summing the exponentiated similarities
over all other antibodies in the batch:

*D* *i* =
∑
*B*

*k* = 1 *;* *k* ∕ = *i*

exp ( *S* *ik* ) *:*

*Step* *4:* *Final* *loss* *calculation.* The total loss is the average of
the negative log likelihood over all positive (same epitope label)
pairs in the batch.
The loss contribution from a single positive pair ( *i* , *j* )
is −
log ^exp^ ^(^ *^S^* *^ij^* ^)^
*D* *i*
.
The final loss averages these terms:

L *contrastive* =
∑
*B*

*i* = | P ( *i* )|

∑
*B*

*i* = ∑

*j* ∈ P ( *i* )

(
− log ^exp^
(

*S* *ij*

)

*D* *i*

)
*:*

Training proceeded for 400 epochs on a single NVIDIA AGPU, requiring approximately 5 h, including inter-epoch evalua-
tions. Model selection was based on AUROC performance on
the validation set (weighted by epitope class size), with the
epoch 280 checkpoint achieving optimal performance.

Histogram generation and pairwise accuracy or Fcalculation
Distributions of antibody pair relationships were visualized and
analyzed using histograms implemented in Python 3 .8.18 with
seaborn 0.13.1. All histograms were generated using probability
density normalization with 30 uniform-width bins. Classification
thresholds were determined differently for AbLang-RBD and
AbLang-PDB versus the pretrained model. For AbLang-RBD
and AbLang-PDB, thresholds were optimized to maximize
balanced accuracy on the validation dataset. The pretrained
model threshold in Figure 4 A was similarly optimized using train
versus validation parameterization, while in Figure 3 A, it was
optimized for maximal balanced accuracy across the complete
dataset (note: this approach overestimates model performance).
For three-category classification, optimal decision boundaries
were determined via grid search across 90,000 threshold combi-
nations (300 × 300 cosine similarity values). The threshold pair
yielding maximum balanced accuracy across all three categories
(overlapping epitopes, non-overlapping epitopes within the
same Pfam, and different Pfams) was selected. Balanced
accuracy was calculated as the mean of individual category
accuracies, in contrast to total accuracy, which can be biased
by class imbalance.

<||WXb23TXrUn3Rxz00yNNr89HV||>epitope, we calculated a weighted average of the SARS-CoV-RBD’s atomic coordinates. The goal was to find the center of
mass for each epitope, where the ‘‘mass’’ of each residue is
determined by its importance for antibody binding.
For each antibody, we first obtained its complete DMS escape
map, which details how every possible mutation to the RBD af-
fects antibody binding. For each residue position on the RBD,
we summed the escape scores of all possible mutations at that
site. This sum, *w* *i* , serves as a weight representing the overall
importance of residue *i* to the antibody’s epitope. We then
used the 3D coordinates ( *x* *i* ) of the ⍺ carbon of each residue
from the SARS-CoV-2 RBD structure (PDB: 8SGU). The final
3D epitope coordinate ( *x* ) was calculated as the weighted
average of these ⍺ carbon positions:

*x* =

∑

*i*

*w* *i* *x* *i*
∑

*i*

*w* *i*

*:*

This method provides a single 3D point for each antibody’s
epitope, allowing for the calculation of simple Euclidean dis-
tances between epitopes for comparison with our model’s pre-
dictions ( Figure S1 C).

Regression analysis
Statistical analyses were performed using SciPy (v.1.10.1) for
correlation calculations and significance testing. ^70^ Spearman’s
rank correlation (spearmanr) and Pearson correlation (pearsonr)
coefficients were calculated for various pairwise comparisons. In
Figure S1 B, the relationship between relative BSA (rBSA) and
training labels was fit using linear regression (scipy.stats.linre-
gress), excluding pairs with labels below 0.5. For correlation an-
alyses in Figures 4 D, 5 B, and 5C, Spearman correlations were
calculated with associated *p* values; *p* values below the numer-
ical precision limit of 64-bit floating point numbers are reported
as *p* < 5e − 3 00.
For Figure 4 B, we calculated the maximum achievable
Spearman correlation ( *ρ* max ) by considering the optimal ranking
scenario where (1) all antibody pairs with a label of − 1 rank below
those with a label of 0 .2, (2) all pairs with label 0 .2 rank below
those with labels of ≥ 0 .5, and (3) pairs with labels between 0 .and 1.0 are perfectly rank ordered. Mean values with 95% con-
fidence intervals were calculated for discrete label categories
( − 1 and 0.2) and for the continuous range of labels ≥ 0 .5 (plotted
at *x* = 0.75). For Figure 4 C, analysis was restricted to pairs with
both predicted cosine similarities and ground-truth labels be-
tween 0 .5 and 1 .0 to assess performance on high-confidence
predictions.

and test sets were subsequently visualized separately to assess
generalization performance.
Clustering analysis was performed using *k* -means with cosine
similarity as the distance metric. The algorithm was initialized
with 12 clusters using the *k* -means++ strategy for greedy
centroid initialization and allowed to run for a maximum of 300 it-
erations. Clustering accuracy was assessed by assigning the
most highly represented epitope class within each cluster as
the cluster’s representative epitope. Antibodies within each clus-
ter were considered accurately clustered if they matched this
epitope and incorrectly clustered otherwise. This approach,
while disadvantaging underrepresented epitopes due to class
imbalance, provides a conservative estimate of clustering
performance.
For visualization clarity, we cycled through three marker
shapes (circles, squares, and triangles), as well as ten distinct
colors.

AbLang-PDB training
The AbLang-PDB model was trained using the architecture
described in the model architecture section, utilizing the
curated structural antibody dataset. During training, we main-
tained the pretrained weights of the base model, modifying
only the QLORA adaptation parameters and the six mixing
layers responsible for cross-chain information integration.
Training employed the AdamW optimizer with a learning rate
of 1e − 5 and a mean squared error loss function, using a batch
size of 1 6.
To address class imbalance in the training data, we imple-
mented a balanced sampling strategy where each epoch pro-
cessed 15,270 antibody pairs, evenly distributed across three
categories: overlapping epitopes, non-overlapping epitopes
within the same Pfam, and pairs targeting different Pfams. While
this approach ensured equal representation of each category
during training, it resulted in more unique pairs from the non-
overlapping-epitope classes being trained on.
Training proceeded for 500 epochs on an NVIDIA AGPU, requiring approximately 36 h, including inter-epoch evalu-
ations. Model selection was based on AUROC performance
comparing training and test sets, with the epoch 240 checkpoint
achieving optimal performance.

<||WXb23TXrUn3Rxz00yNNr89HV||>Receiver operating characteristic, precision-recall, and
F1-score calculation
Model performance was evaluated using multiple complemen-
tary metrics implemented through scikit-learn. For receiver
operating characteristic (ROC) analysis, we calculated true
positive and false positive rates across 2,001 equally spaced
thresholds spanning the range of possible prediction values
(cosine similarity from − 1 to 1 for model predictions and
from 0 to 1 for sequence identity comparisons). The area
under the ROC curve was computed using scikit-learn’s
trapezoidal rule implementation. For AbLang-RBD, AUROC
values were calculated separately for each of the 12 epitope
classes and combined using a weighted average based on
class size. For AbLang-PDB, the calculation used a binary
classification scheme where overlapping-epitope pairs consti-
tuted the positive class and non-overlapping pairs the nega-
tive class.
Precision-recall characteristics were assessed using scikit-
learn’s precision_recall_curve and average_precision_score
functions. For F1-score calculations, we utilized the previously
determined optimal threshold that maximized balanced
accuracy. In the case of Pfam classification, F1-scores were
calculated considering all antibody pairs targeting the same
Pfam as positives, regardless of their specific epitope overlap
status.

LIBRA-seq dataset curation
Antibody sequence datasets for 8ANC195-like antibody mining
came from previous in-house LIBRA-seq
experiments. ^53^

LIBRA-seq is a high-throughput technology that enables simul-
taneous identification of antigen specificity and BCR sequences
at single-cell resolution. In this approach, B cells are exposed
to oligonucleotide-barcoded antigens, allowing quantitative
assessment of antigen binding through unique molecular identi-
fiers during subsequent single-cell sequencing. From these ex-
periments, 7,056 class-switched antibody sequences were
compiled using peripheral blood mononuclear cells (PBMCs)
from persons living with HIV-1.
The dataset comprised 21 LIBRA-seq experiments where
antigen-specific B cells were isolated from PBMCs using fluo-
rescence-activated cell sorting (FACS). While this experi-
mental design enriched for HIV-1-specific antibodies in the
dataset, the majority of sequences are not expected to be
HIV-1 specific. Analysis included only functional, single-cell
records from 10 × Genomics VDJ sequencing where cells
had undergone FACS using PE-labeled antigens, including
at least one HIV Env protein and one unrelated control anti-
gen. Each antibody was assigned a unique identifier contain-
ing a 4-digit sequencing run prefix, with most run prefixes cor-
responding to unique donors except for runs 2723 and 3514,
which both originated from donor 45 (source of VRC01). ^53^ ^,^ ^71^

specificity toward positive control and negative control anti-
gens, enabling unbiased identification of antibodies with po-
tential epitope overlap based solely on sequence features
learned by the AbLang-PDB model.

Antibody production
Antibody heavy and light chains were synthesized as cDNA by
Twist Bioscience or Genscript. Variable genes were inserted
into either bicistronic plasmids encoding the constant regions
of the H chain and either the kappa or lambda light chain or
into separate heavy- and light-chain plasmids. mAbs made in
house were transiently expressed using the ExpiFectamine
transfection reagent (Thermo Fisher Scientific) in Expi293F cells
in FreeStyle F17 media supplemented with 0 .1% poloxamer
188 and 20% 4 mM L-glutamine (Thermo Fisher Scientific).
Transfected cultures were incubated shaking for 5 days at
37 ◦ C with 8% CO 2 saturation. After 5 days, cultures were har-
vested and centrifuged at a minimum of 4,000 rpm for 20 min.
Supernatant was then filtered with Nalgene Rapid-Flow dispos-
able filter units with a polyethersulfone (PES) membrane (0.or 0 .22 μ m). Filtrate was run over phosphate-buffered saline
(PBS) equilibrated columns containing protein A resin. Columns
were then washed with PBS, and purified antibodies were
eluted using 10 mL of 100 mM glycine HCL at pH 2 .7 into
1 mL of 1 M Tris-HCl (pH 8). These were then buffer exchanged
into PBS. The remaining mAbs were synthesized by Genscript
in their 10 mL TurboCHO high-throughput antibody expression
system.

<||WXb23TXrUn3Rxz00yNNr89HV||>intermolecular disulfide bond between gp120 and gp41 (A501C
and T605C), a trimer-stabilizing mutation (I559P), a truncated
gp41 transmembrane region at position 664, and an I201C/
A433C mutation to inhibit CD4-induced movement of Env. Addi-
tionally, a flexible serine-glycine linker was inserted between
gp120 and gp41 (positions 507 and 5 12) to create single-chain
constructs. ^77^

HIV-1 Env proteins were expressed in a highly similar fashion
but with the following caveats. Post-culture and centrifugation,
the filtered supernatant was applied to an affinity column of
agarose-bound *Galanthus* *nivalis* lectin (Vector Laboratories) at
4 ◦ C. After washing with PBS, proteins were eluted with 30 mL
of 1 M methyl- α -D-mannopyranoside. The eluate was buffer
exchanged three times into PBS and concentrated using either
30 or 100 kDa Amicon Ultra centrifugal filter units.
Final purification was achieved by size-exclusion chromatog-
raphy using either a Superose 6 Increase 10/300 GL or Superdex
200 Increase 10/300 GL column on an AKTA FPLC system.
Fractions corresponding to correctly folded trimeric Env proteins
were collected and validated by SDS-PAGE for molecular
weight determination and by ELISA for antigenicity using
Env-specific mAbs.

Indirect ELISA
In a 96-well plate, 100 μ L of antigen was coated at 2 μ g/mL over-
night at 4 ◦ C. The plates were then washed three times with PBS
supplemented with 0.05% Tween 20 (PBS-T) and blocked using
5% bovine serum albumin in PBS. Plates were incubated for 1 h
at room temperature and then washed three times using PBS-T.
Primary antibodies were diluted in 1% bovine serum albumin in
PBS-T starting at 10 μ g/mL with a 1:5 dilution. After incubating
at room temperature for 1 h and washing with PBS-T, 100 μ L
of goat anti-human immunoglobulin G (IgG) conjugated to perox-
idase was added at a 1:10,000 dilution in 1% bovine serum albu-
min in PBS-T. These were incubated for 1 h at room temperature,
washed three times with PBS-T, and then developed using
3,3 ^′^ ,5,5 ^′^ tetramethylbenzidine dihydrochloride (TMB) substrate.
Plates were developed for 10 min at room temperature and
were then stopped using 1 N sulfuric acid. Absorbance was
then measured at 450 nm.

the percentage change in binding relative to the binding of an
antibody when no competitor is present.

Structural representation of HIV reference antibodies
A composite image of VRC01, 8ANC195, and PG9 binding
BG505 Env was generated by first loading PDB: 5VJ6 into
open-source PyMOL Schrodinger, v.2.4.0. ^56^ ^,^ ^65^ The antibody-an-
tigen complex was represented as a surface, and PG9 was
colored wheat, 8ANC195 light green, gp120 light gray, and
gp41 dark gray. PDB: 8VGW was then loaded into PyMOL,
and the gp120 structures from one protomer were aligned to
that of one gp120 protomer in 5VJ6. VRC01 was then colored
pink and shown as a surface without visualization of the Env pro-
tein present in its native complex. Finally, ray tracing was per-
formed with default parameters.

Benchmarking baseline models
AUROC, average precision, and F1-scores were calculated
as described previously. For the following baseline models,
the final residue-level hidden state was averaged over just the
heavy chain to generate embeddings: AbLang-Heavy ^28^ and
AntiBERTy. ^27^ For the following models, the hidden state was aver-
aged over both the heavy and light chains: AbLang2, ^29^ IgBERT, ^30^

BALM, ^31^ ESM-2, ^25^ and Parapred. ^26^ For Parapred, this averaging
was restricted to CDR residues, with each CDR fed into the model
separately, as is standard for this model. ^26^ For ESM-2, the 650 M
parameter model was used, and the heavy and light chains were
fed simultaneously, separated by two classification (CLS) tokens
as done in Burbach and Briney. ^31^ For AbLang-Pre, we separately
averaged the residue-level embeddings from AbLang-Heavy and
AbLang-Light ^28^ and concatenated them sequentially.

RESOURCE AVAILABILITY

Lead contact
Further information and requests for resources and reagents should be
directed to and will be fulfilled by the lead contact, Ivelin S. Georgiev ( [ivelin.](mailto:ivelin.georgiev@vanderbilt.edu)
[georgiev@vanderbilt.edu](mailto:ivelin.georgiev@vanderbilt.edu)[).](mailto:ivelin.georgiev@vanderbilt.edu)

Materials availability
Materials will be made available upon request under a completed materials
transfer agreement (MTA).

Data and code availability

<||WXb23TXrUn3Rxz00yNNr89HV||>ACKNOWLEDGMENTS

We thank Perry Wasdin for his help in curating antigen-specific training
sets that helped to interrogate the efficacy of different models and loss
functions and Alexandra Abu-Shmais for her insights on drawing conclusions
from pairwise antibody comparisons. We additionally thank Andrea Shiakolas,
Ian Setliff, Kelsey Pilewski, Rohit Venkat, and Lauren Walker for LIBRA-seq
data. This research was funded, in part, by the Advanced Research Projects
Agency for Health (ARPA-H 1AY2AX000077), NIH R01AI175245, and the
G. Harold and Leila Y. Mathers Charitable Foundation (MF-2107-01851).
The funders had no role in the conceptualization or execution of any studies
or drafting of the manuscript. The views and conclusions contained in
this document are those of the authors and should not be interpreted as
representing the official policies, either expressed or implied, of the US
government.

AUTHOR CONTRIBUTIONS

Conceptualization, C.M.H. and I.S.G.; data curation, C.M.H.; formal analysis,
C.M.H. and T.M.M.; software, C.M.H. and T.M.M.; methodology, C.M.H. and
A.K.J.; investigation, C.M.H., I.S.G., A.K.J., P.A., T.M.M., and P.J.J.; visualiza-
tion, C.M.H., I.S.G., and A.K.J.; writing – original draft, C.M.H.; writing – re-
view & editing, C.M.H., I.S.G., A.K.J., P.A., T.M.M., and P.J.J.; funding acqui-
sition, I.S.G.; project administration, I.S.G.; supervision, I.S.G.; validation,
A.K.J. and P.A.; resources, P.J.J.

DECLARATION OF INTERESTS

I.S.G. is listed as an inventor on patents filed describing antibodies character-
ized here. I.S.G. is listed as an inventor on the patent applications for the
LIBRA-seq technology. I.S.G. is a co-founder of AbSeek Bio. I.S.G. has served
as a consultant for Sanofi. The Georgiev laboratory at VUMC has received un-
related funding from Merck and Takeda Pharmaceuticals.

SUPPLEMENTAL INFORMATION

Supplemental information can be found online at [https://doi.org/10.1016/j.](https://doi.org/10.1016/j.patter.2025.101419)
[patter.2025.101419](https://doi.org/10.1016/j.patter.2025.101419)[.](https://doi.org/10.1016/j.patter.2025.101419)

Received: March 19, Revised: September 16, Accepted: October 14, Published: November 13, REFERENCES

1. Lu, R.-M., Hwang, Y.-C., Liu, I.-J., Lee, C.-C., Tsai, H.-Z., Li, H.-J., and Wu,H.-C. (2020). Development of therapeutic antibodies for the treatment of
diseases. J. Biomed. Sci. *27* , 1 . [https://doi.org/10.1186/s12929-019-](https://doi.org/10.1186/s12929-019-0592-z)
[0592-z](https://doi.org/10.1186/s12929-019-0592-z)[.](https://doi.org/10.1186/s12929-019-0592-z)

1. Chames, P., Van Regenmortel, M., Weiss, E., and Baty, D. (2009).Therapeutic antibodies: successes, limitations and hopes for the future.
Br. J. Pharmacol. *157* , 220–233. [https://doi.org/10.1111/j.1476-5381.](https://doi.org/10.1111/j.1476-5381.2009.00190.x)
[2 009.00190.x](https://doi.org/10.1111/j.1476-5381.2009.00190.x)[.](https://doi.org/10.1111/j.1476-5381.2009.00190.x)

1. Mahomed, S. (2024). Broadly neutralizing antibodies for HIV prevention: acomprehensive review and future perspectives. Clin. Microbiol. Rev. *37* ,
e00152-22. [https://doi.org/10.1128/cmr.00152-22](https://doi.org/10.1128/cmr.00152-22)[.](https://doi.org/10.1128/cmr.00152-22)

1. Labrijn, A.F., Janmaat, M.L., Reichert, J.M., and Parren, P.W.H.I. (2019).Bispecific antibodies: a mechanistic review of the pipeline. Nat. Rev.
Drug Discov. *18* , 585–608. [https://doi.org/10.1038/s41573-019-0028-1](https://doi.org/10.1038/s41573-019-0028-1)[.](https://doi.org/10.1038/s41573-019-0028-1)

1. Saphire, E.O., Schendel, S.L., Fusco, M.L., Gangavarapu, K., Gunn, B.M.,Wec, A.Z., Halfmann, P.J., Brannan, J.M., Herbert, A.S., Qiu, X., et al.
(2018). Systematic analysis of monoclonal antibodies against Ebola virus
GP defines features that contribute to protection. Cell *174* , 938–
9 52.e13. [https://doi.org/10.1016/j.cell.2018.07.033](https://doi.org/10.1016/j.cell.2018.07.033)[.](https://doi.org/10.1016/j.cell.2018.07.033)

neutralizing and protective human antibodies against SARS-CoV-2.
Nature *584* , 443–449. [https://doi.org/10.1038/s41586-020-2548-6](https://doi.org/10.1038/s41586-020-2548-6)[.](https://doi.org/10.1038/s41586-020-2548-6)

1. Olsen, T.H., Abanades, B., Moal, I.H., and Deane, C.M. (2023). KA-Search,a method for rapid and exhaustive sequence identity search of known
antibodies. Sci. Rep. *13* , 1 1612. [https://doi.org/10.1038/s41598-023-](https://doi.org/10.1038/s41598-023-38108-7)
[38108-7](https://doi.org/10.1038/s41598-023-38108-7)[.](https://doi.org/10.1038/s41598-023-38108-7)

1. Chen, E.C., Gilchuk, P., Zost, S.J., Suryadevara, N., Winkler, E.S., Cabel,C.R., Binshtein, E., Chen, R.E., Sutton, R.E., Rodriguez, J., et al. (2021).
Convergent antibody responses to the SARS-CoV-2 spike protein in
convalescent and vaccinated individuals. Cell Rep. *36* , 1 09604. [https://](https://doi.org/10.1016/j.celrep.2021.109604)
[doi.org/10.1016/j.celrep.2021.109604](https://doi.org/10.1016/j.celrep.2021.109604)[.](https://doi.org/10.1016/j.celrep.2021.109604)

1. Chen, E.C., Gilchuk, P., Zost, S.J., Ilinykh, P.A., Binshtein, E., Huang, K.,Myers, L., Bonissone, S., Day, S., Kona, C.R., et al. (2023). Systematic
analysis of human antibody response to ebolavirus glycoprotein shows
high prevalence of neutralizing public clonotypes. Cell Rep. *42* , 1 12370.
[https://doi.org/10.1016/j.celrep.2023.112370](https://doi.org/10.1016/j.celrep.2023.112370)[.](https://doi.org/10.1016/j.celrep.2023.112370)

1 0. Abu-Shmais, A.A., Vukovich, M.J., Wasdin, P.T., Suresh, Y.P., Marinov,
T.M., Rush, S.A., Gillespie, R.A., Sankhala, R.S., Choe, M., Joyce, M.G.,
et al. (2024). Antibody sequence determinants of viral antigen specificity.
mBio *15* , e01560-24. [https://doi.org/10.1128/mbio.01560-24](https://doi.org/10.1128/mbio.01560-24)[.](https://doi.org/10.1128/mbio.01560-24)

1 1. Chinery, L., Wahome, N., Moal, I., and Deane, C.M. (2023). Paragraph—
antibody paratope prediction using graph neural networks with minimal
feature vectors. Bioinformatics *39* , btac732. [https://doi.org/10.1093/bio-](https://doi.org/10.1093/bioinformatics/btac732)
[informatics/btac732](https://doi.org/10.1093/bioinformatics/btac732)[.](https://doi.org/10.1093/bioinformatics/btac732)

1 2. Olsen, T.H., Boyles, F., and Deane, C.M. (2022). Observed Antibody
Space: A diverse database of cleaned, annotated, and translated unpaired
and paired antibody sequences. Protein Sci. *31* , 141–146. [https://doi.org/](https://doi.org/10.1002/pro.4205)
[1 0.1002/pro.4205](https://doi.org/10.1002/pro.4205)[.](https://doi.org/10.1002/pro.4205)

1 3. Richardson, E., Galson, J.D., Kellam, P., Kelly, D.F., Smith, S.E., Palser, A.,
Watson, S., and Deane, C.M. (2021). A computational method for immune
repertoire mining that identifies novel binders from different clonotypes,
demonstrated by identifying anti-pertussis toxoid antibodies. mAbs *13* ,
1 869406. [https://doi.org/10.1080/19420862.2020.1869406](https://doi.org/10.1080/19420862.2020.1869406)[.](https://doi.org/10.1080/19420862.2020.1869406)

1 4. Wong, W.K., Robinson, S.A., Bujotzek, A., Georges, G., Lewis, A.P., Shi,
J., Snowden, J., Taddese, B., and Deane, C.M. (2021). Ab-Ligity: identi-
fying sequence-dissimilar antibodies that bind to the same epitope.
mAbs *13* , 1873478. [https://doi.org/10.1080/19420862.2021.1873478](https://doi.org/10.1080/19420862.2021.1873478)[.](https://doi.org/10.1080/19420862.2021.1873478)

1 5. Robinson, S.A., Raybould, M.I.J., Schneider, C., Wong, W.K., Marks, C., and
Deane, C.M. (2021). Epitope profiling using computational structural model-
ling demonstrated on coronavirus-binding antibodies. PLoS Comput. Biol.
*17* , e1009675. [https://doi.org/10.1371/journal.pcbi.1009675](https://doi.org/10.1371/journal.pcbi.1009675)[.](https://doi.org/10.1371/journal.pcbi.1009675)

1 6. [Spoendlin,](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref16)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref16)[F.C.,](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref16)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref16)[Abanades,](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref16)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref16)[B.,](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref16)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref16)[Raybould,](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref16)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref16)[M.I.J.,](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref16)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref16)[Wong,](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref16)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref16)[W.K.,](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref16)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref16)[Georges,](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref16)
[G., and Deane, C.M. (2023). Improved computational epitope profiling us-](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref16)
[ing structural models identifies a broader diversity of antibodies that bind](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref16)
[to the same epitope. Front. Mol. Biosci.](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref16)[*10*](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref16)[, 1237621](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref16)[.](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref16)

1 7. Wang, Y., Yuan, M., Lv, H., Peng, J., Wilson, I.A., and Wu, N.C. (2022). A
large-scale systematic survey reveals recurring molecular features of pub-
lic antibody responses to SARS-CoV-2. Immunity *55* , 1105–1117.e4.
[https://doi.org/10.1016/j.immuni.2022.03.019](https://doi.org/10.1016/j.immuni.2022.03.019)[.](https://doi.org/10.1016/j.immuni.2022.03.019)

1 8. Strasser, J., de Jong, R.N., Beurskens, F.J., Wang, G., Heck, A.J.R.,
Schuurman, J., Parren, P.W.H.I., Hinterdorfer, P., and Preiner, J. (2019).
Unraveling the Macromolecular Pathways of IgG Oligomerization and
Complement Activation on Antigenic Surfaces. Nano Lett. *19* , 4787–
4 796. [https://doi.org/10.1021/acs.nanolett.9b02220](https://doi.org/10.1021/acs.nanolett.9b02220)[.](https://doi.org/10.1021/acs.nanolett.9b02220)

1 9. Goldberg, B.S., and Ackerman, M.E. (2020). Antibody-mediated comple-
ment activation in pathology and protection. Immunol. Cell Biol. *98* ,
305–317. [https://doi.org/10.1111/imcb.12324](https://doi.org/10.1111/imcb.12324)[.](https://doi.org/10.1111/imcb.12324)

2 0. Huang, J., Kang, B.H., Ishida, E., Zhou, T., Griesman, T., Sheng, Z., Wu, F.,
Doria-Rose, N.A., Zhang, B., McKee, K., et al. (2016). Identification of a
CD4-Binding-Site Antibody to HIV that Evolved Near-Pan Neutralization
Breadth. Immunity *45* , 1108–1121. [https://doi.org/10.1016/j.immuni.](https://doi.org/10.1016/j.immuni.2016.10.027)
[2 016.10.027](https://doi.org/10.1016/j.immuni.2016.10.027)[.](https://doi.org/10.1016/j.immuni.2016.10.027)

<||WXb23TXrUn3Rxz00yNNr89HV||>and computational methods. mAbs *13* , 1895540. [https://doi.org/10.1080/](https://doi.org/10.1080/19420862.2021.1895540)
[1 9420862.2021.1895540](https://doi.org/10.1080/19420862.2021.1895540)[.](https://doi.org/10.1080/19420862.2021.1895540)

2 2. Liu, C., Denzler, L., Chen, Y., Paige, B., and Martin, A. (2024). AsEP:
Benchmarking Deep Learning Methods for Antibody-specific Epitope
Prediction. [https://doi.org/10.5281/ZENODO.11495514](https://doi.org/10.5281/ZENODO.11495514)[.](https://doi.org/10.5281/ZENODO.11495514)

2 3. Eguchi, R.R., Choe, C.A., and Huang, P.-S. (2022). Ig-VAE: Generative
modeling of protein structure by direct 3D coordinate generation. PLoS
Comput. Biol. *18* , e1010271. [https://doi.org/10.1371/journal.pcbi.1010271](https://doi.org/10.1371/journal.pcbi.1010271)[.](https://doi.org/10.1371/journal.pcbi.1010271)

2 4. Shan, S., Luo, S., Yang, Z., Hong, J., Su, Y., Ding, F., Fu, L., Li, C., Chen, P.,
Ma, J., et al. (2022). Deep learning guided optimization of human antibody
against SARS-CoV-2 variants with broad neutralization. Proc. Natl. Acad.
Sci. USA *119* , e2122954119. [https://doi.org/10.1073/pnas.2122954119](https://doi.org/10.1073/pnas.2122954119)[.](https://doi.org/10.1073/pnas.2122954119)

2 5. Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., Smetanin, N., Verkuil, R.,
Kabeli, O., Shmueli, Y., et al. (2023). Evolutionary-scale prediction of
atomic-level protein structure with a language model. Science *379* ,
1123–1130. [https://doi.org/10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)[.](https://doi.org/10.1126/science.ade2574)

2 6. Liberis, E., Veli  ckovi  c, P., Sormanni, P., Vendruscolo, M., and Lio` , P.
(2018). Parapred: antibody paratope prediction using convolutional and
recurrent neural networks. Bioinformatics *34* , 2944–2950. [https://doi.](https://doi.org/10.1093/bioinformatics/bty305)
[org/10.1093/bioinformatics/bty305](https://doi.org/10.1093/bioinformatics/bty305)[.](https://doi.org/10.1093/bioinformatics/bty305)

2 7. Ruffolo, J.A., Gray, J.J., and Sulam, J. (2021). Deciphering antibody affinity
maturation with language models and weakly supervised learning.
Preprint at arXiv. [https://doi.org/10.48550/arXiv.2112.07782](https://doi.org/10.48550/arXiv.2112.07782)[.](https://doi.org/10.48550/arXiv.2112.07782)

2 8. Olsen, T.H., Moal, I.H., and Deane, C.M. (2022). AbLang: an antibody lan-
guage model for completing antibody sequences. Bioinform. Adv. *2* ,
vbac046. [https://doi.org/10.1093/bioadv/vbac046](https://doi.org/10.1093/bioadv/vbac046)[.](https://doi.org/10.1093/bioadv/vbac046)

2 9. Olsen, T.H., Moal, I.H., and Deane, C.M. (2024). Addressing the antibody
germline bias and its effect on language models for improved antibody
design. Bioinformatics *40* , btae618. [https://doi.org/10.1093/bioinformat-](https://doi.org/10.1093/bioinformatics/btae618)
[ics/btae618](https://doi.org/10.1093/bioinformatics/btae618)[.](https://doi.org/10.1093/bioinformatics/btae618)

3 0. Kenlay, H., Dreyer, F.A., Kovaltsuk, A., Miketa, D., Pires, D., and Deane,
C.M. (2024). Large scale paired antibody language models. PLoS
Comput. Biol. *20* , e1012646. [https://doi.org/10.1371/journal.pcbi.1012646](https://doi.org/10.1371/journal.pcbi.1012646)[.](https://doi.org/10.1371/journal.pcbi.1012646)

3 1. Burbach, S.M., and Briney, B. (2024). Improving antibody language
models with native pairing. Patterns *5* , 100967. [https://doi.org/10.1016/j.](https://doi.org/10.1016/j.patter.2024.100967)
[patter.2024.100967](https://doi.org/10.1016/j.patter.2024.100967)[.](https://doi.org/10.1016/j.patter.2024.100967)

3 2. Bepler, T., and Berger, B. (2021). Learning the Protein Language:
Evolution, Structure and Function. Cell Syst. *12* , 654–669.e3. [https://doi.](https://doi.org/10.1016/j.cels.2021.05.017)
[org/10.1016/j.cels.2021.05.017](https://doi.org/10.1016/j.cels.2021.05.017)[.](https://doi.org/10.1016/j.cels.2021.05.017)

3 3. Dunbar, J., Krawczyk, K., Leem, J., Baker, T., Fuchs, A., Georges, G., Shi, J.,
and Deane, C.M. (2014). SAbDab: the structural antibody database. Nucleic
Acids Res. *42* , D1140–D1146. [https://doi.org/10.1093/nar/gkt1043](https://doi.org/10.1093/nar/gkt1043)[.](https://doi.org/10.1093/nar/gkt1043)

3 4. Schneider, C., Raybould, M.I.J., and Deane, C.M. (2022). SAbDab in the
age of biotherapeutics: updates including SAbDab-nano, the nanobody
structure tracker. Nucleic Acids Res. *50* , D1368–D1372. [https://doi.org/](https://doi.org/10.1093/nar/gkab1050)
[1 0.1093/nar/gkab1050](https://doi.org/10.1093/nar/gkab1050)[.](https://doi.org/10.1093/nar/gkab1050)

3 5. Punta, M., Coggill, P.C., Eberhardt, R.Y., Mistry, J., Tate, J., Boursnell, C.,
Pang, N., Forslund, K., Ceric, G., Clements, J., et al. (2012). The Pfam pro-
tein families database. Nucleic Acids Res. *40* , D290–D301. [https://doi.org/](https://doi.org/10.1093/nar/gkr1065)
[1 0.1093/nar/gkr1065](https://doi.org/10.1093/nar/gkr1065)[.](https://doi.org/10.1093/nar/gkr1065)

3 6. Raybould, M.I.J., Kovaltsuk, A., Marks, C., and Deane, C.M. (2021). CoV-
AbDab: the Coronavirus Antibody Database. Bioinformatics *37* , 734–735.
[https://doi.org/10.1093/bioinformatics/btaa739](https://doi.org/10.1093/bioinformatics/btaa739)[.](https://doi.org/10.1093/bioinformatics/btaa739)

3 7. Cao, Y., Jian, F., Wang, J., Yu, Y., Song, W., Yisimayi, A., Wang, J., An, R.,
Chen, X., Zhang, N., et al. (2023). Imprinted SARS-CoV-2 humoral immu-
nity induces convergent Omicron RBD evolution. Nature *614* , 521–529.
[https://doi.org/10.1038/s41586-022-05644-7](https://doi.org/10.1038/s41586-022-05644-7)[.](https://doi.org/10.1038/s41586-022-05644-7)

3 8. Cao, Y., Yisimayi, A., Jian, F., Song, W., Xiao, T., Wang, L., Du, S., Wang,
J., Li, Q., Chen, X., et al. (2022). BA.2.12.1, BA.4 and BA.5 escape anti-
bodies elicited by Omicron infection. Nature *608* , 593–602. [https://doi.](https://doi.org/10.1038/s41586-022-04980-y)
[org/10.1038/s41586-022-04980-y](https://doi.org/10.1038/s41586-022-04980-y)[.](https://doi.org/10.1038/s41586-022-04980-y)

4 0. Oord, A. van den, Li, Y., and Vinyals, O. (2018). Representation Learning
with Contrastive Predictive Coding. Preprint at arXiv. [https://doi.org/10.](https://doi.org/10.48550/arXiv.1807.03748)
[48550/arXiv.1807.03748](https://doi.org/10.48550/arXiv.1807.03748)[.](https://doi.org/10.48550/arXiv.1807.03748)

4 1. Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P.,
Maschinot, A., Liu, C., and Krishnan, D. (2020). Supervised Contrastive
Learning. Preprint at arXiv. [https://doi.org/10.48550/arXiv.2004.11362](https://doi.org/10.48550/arXiv.2004.11362)[.](https://doi.org/10.48550/arXiv.2004.11362)

4 2. [Maaten,](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref42)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref42)[L.](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref42)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref42)[van](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref42)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref42)[der,](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref42)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref42)[and](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref42)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref42)[Hinton,](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref42)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref42)[G.](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref42)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref42)[(2008).](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref42)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref42)[Visualizing](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref42)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref42)[Data](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref42)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref42)[using](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref42)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref42)[t-SNE.](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref42)
[J. Mach. Learn. Res.](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref42)[*9*](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref42)[, 2579–2605](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref42)[.](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref42)

4 3. Lloyd, S. (1982). Least squares quantization in PCM. IEEE Trans. Inf.
Theor. *28* , 129–137. [https://doi.org/10.1109/TIT.1982.1056489](https://doi.org/10.1109/TIT.1982.1056489)[.](https://doi.org/10.1109/TIT.1982.1056489)

4 4. Deshpande, A., Harris, B.D., Martinez-Sobrido, L., Kobie, J.J., and Walter,
M.R. (2021). Epitope Classification and RBD Binding Properties of
Neutralizing Antibodies Against SARS-CoV-2 Variants of Concern. Front.
Immunol. *12* , 691715. [https://doi.org/10.3389/fimmu.2021.691715](https://doi.org/10.3389/fimmu.2021.691715)[.](https://doi.org/10.3389/fimmu.2021.691715)

4 5. Sankhala, R.S., Dussupt, V., Chen, W.-H., Bai, H., Martinez, E.J., Jensen,
J.L., Rees, P.A., Hajduczki, A., Chang, W.C., Choe, M., et al. (2024).
Antibody targeting of conserved sites of vulnerability on the SARS-CoV-
2 spike receptor-binding domain. Structure *32* , 131–147.e7. [https://doi.](https://doi.org/10.1016/j.str.2023.11.015)
[org/10.1016/j.str.2023.11.015](https://doi.org/10.1016/j.str.2023.11.015)[.](https://doi.org/10.1016/j.str.2023.11.015)

4 6. Berman, H., Henrick, K., and Nakamura, H. (2003). Announcing the world-
wide Protein Data Bank. Nat. Struct. Biol. *10* , 980. [https://doi.org/10.1038/](https://doi.org/10.1038/nsb1203-980)
[nsb1203-980](https://doi.org/10.1038/nsb1203-980)[.](https://doi.org/10.1038/nsb1203-980)

4 7. Berman, H., Henrick, K., Nakamura, H., and Markley, J.L. (2007). The
worldwide Protein Data Bank (wwPDB): ensuring a single, uniform archive
of PDB data. Nucleic Acids Res. *35* , D301–D303. [https://doi.org/10.1093/](https://doi.org/10.1093/nar/gkl971)
[nar/gkl971](https://doi.org/10.1093/nar/gkl971)[.](https://doi.org/10.1093/nar/gkl971)

4 8. wwPDB Consortium (2019). Protein Data Bank: the single global archive
for 3D macromolecular structure data. Nucleic Acids Res. *47* , D520–
D528. [https://doi.org/10.1093/nar/gky949](https://doi.org/10.1093/nar/gky949)[.](https://doi.org/10.1093/nar/gky949)

4 9. Scheid, J.F., Mouquet, H., Ueberheide, B., Diskin, R., Klein, F., Oliveira,
T.Y.K., Pietzsch, J., Fenyo, D., Abadir, A., Velinzon, K., et al. (2011).
Sequence and Structural Convergence of Broad and Potent HIV
Antibodies That Mimic CD4 Binding. Science *333* , 1633–1637. [https://](https://doi.org/10.1126/science.1207227)
[doi.org/10.1126/science.1207227](https://doi.org/10.1126/science.1207227)[.](https://doi.org/10.1126/science.1207227)

5 0. Scharf, L., Scheid, J.F., Lee, J.H., West, A.P., Chen, C., Gao, H.,
Gnanapragasam, P.N.P., Mares, R., Seaman, M.S., Ward, A.B., et al.
(2014). Antibody 8ANC195 Reveals a Site of Broad Vulnerability on the
HIV-1 Envelope Spike. Cell Rep. *7* , 785–795. [https://doi.org/10.1016/j.cel-](https://doi.org/10.1016/j.celrep.2014.04.001)
[rep.2014.04.001](https://doi.org/10.1016/j.celrep.2014.04.001)[.](https://doi.org/10.1016/j.celrep.2014.04.001)

5 1. Scharf, L., Wang, H., Gao, H., Chen, S., McDowall, A.W., and Bjorkman,
P.J. (2015). Broadly Neutralizing Antibody 8ANC195 Recognizes Closed
and Open States of HIV-1 Env. Cell *162* , 1379–1390. [https://doi.org/10.](https://doi.org/10.1016/j.cell.2015.08.035)
[1016/j.cell.2015.08.035](https://doi.org/10.1016/j.cell.2015.08.035) .

5 2. Griffith, S.A., and McCoy, L.E. (2021). To bnAb or Not to bnAb: Defining
Broadly Neutralising Antibodies Against HIV-1. Front. Immunol. *12* ,
7 08227. [https://doi.org/10.3389/fimmu.2021.708227](https://doi.org/10.3389/fimmu.2021.708227)[.](https://doi.org/10.3389/fimmu.2021.708227)

5 3. Setliff, I., Shiakolas, A.R., Pilewski, K.A., Murji, A.A., Mapengo, R.E.,
Janowska, K., Richardson, S., Oosthuysen, C., Raju, N., Ronsard, L.,
et al. (2019). High-Throughput Mapping of B Cell Receptor Sequences
to Antigen Specificity. Cell *179* , 1636–1646.e15. [https://doi.org/10.1016/](https://doi.org/10.1016/j.cell.2019.11.003)
[j.cell.2019.11.003](https://doi.org/10.1016/j.cell.2019.11.003)[.](https://doi.org/10.1016/j.cell.2019.11.003)

5 4. Walker, L.M., Shiakolas, A.R., Venkat, R., Liu, Z.A., Wall, S., Raju, N.,
Pilewski, K.A., Setliff, I., Murji, A.A., Gillespie, R., et al. (2022). High-
Throughput B Cell Epitope Determination by Next-Generation Sequencing.
Front. Immunol. *13* , 855772. [https://doi.org/10.3389/fimmu.2022.855772](https://doi.org/10.3389/fimmu.2022.855772)[.](https://doi.org/10.3389/fimmu.2022.855772)

5 5. Zhou, T., Georgiev, I., Wu, X., Yang, Z.-Y., Dai, K., Finzi, A., Kwon, Y.D.,
Scheid, J.F., Shi, W., Xu, L., et al. (2010). Structural basis for broad and
potent neutralization of HIV-1 by antibody VRC01. Science *329* ,
811–817. [https://doi.org/10.1126/science.1192819](https://doi.org/10.1126/science.1192819)[.](https://doi.org/10.1126/science.1192819)

<||WXb23TXrUn3Rxz00yNNr89HV||>5 7. Jespers, L.S., Roberts, A., Mahler, S.M., Winter, G., and Hoogenboom,
H.R. (1994). Guiding the selection of human antibodies from phage display
repertoires to a single epitope of an antigen. Biotechnology *12* , 899–903.
[https://doi.org/10.1038/nbt0994-899](https://doi.org/10.1038/nbt0994-899)[.](https://doi.org/10.1038/nbt0994-899)

5 8. Sundararajan, M., Taly, A., and Yan, Q. (2017). Axiomatic Attribution
for Deep Networks. Preprint at arXiv. [https://doi.org/10.48550/arXiv.](https://doi.org/10.48550/arXiv.1703.01365)
[1 703.01365](https://doi.org/10.48550/arXiv.1703.01365)[.](https://doi.org/10.48550/arXiv.1703.01365)

5 9. Zielezinski, A. (2025). pfam_scan [Computer software]. GitHub. [https://](https://github.com/aziele/pfam_scan)
[github.com/aziele/pfam_scan](https://github.com/aziele/pfam_scan) .

6 0. Shrake, A., and Rupley, J.A. (1973). Environment and exposure to solvent
of protein atoms. Lysozyme and insulin. J. Mol. Biol. *79* , 351–371. [https://](https://doi.org/10.1016/0022-2836(73)90011-9)
[doi.org/10.1016/0022-2836(73)90011-9](https://doi.org/10.1016/0022-2836(73)90011-9)[.](https://doi.org/10.1016/0022-2836(73)90011-9)

6 1. Kabsch, W., and Sander, C. (1983). Dictionary of protein secondary struc-
ture: Pattern recognition of hydrogen-bonded and geometrical features.
Biopolymers *22* , 2577–2637. [https://doi.org/10.1002/bip.360221211](https://doi.org/10.1002/bip.360221211)[.](https://doi.org/10.1002/bip.360221211)

6 2. Touw, W.G., Baakman, C., Black, J., te Beek, T.A.H., Krieger, E., Joosten,
R.P., and Vriend, G. (2015). A series of PDB-related databanks for
everyday needs. Nucleic Acids Res. *43* , D364–D368. [https://doi.org/10.](https://doi.org/10.1093/nar/gku1028)
[1093/nar/gku1028](https://doi.org/10.1093/nar/gku1028)[.](https://doi.org/10.1093/nar/gku1028)

6 3. [Henikoff,](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref63)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref63)[S.,](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref63)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref63)[and](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref63)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref63)[Henikoff,](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref63)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref63)[J.G.](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref63)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref63)[(1992).](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref63)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref63)[Amino](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref63)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref63)[acid](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref63)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref63)[substitution](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref63)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref63)[matrices](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref63)
[from protein blocks. Proc. Natl. Acad. Sci. USA](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref63)[*89*](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref63)[, 10915–10919](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref63)[.](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref63)

6 4. Needleman, S.B., and Wunsch, C.D. (1970). A general method applicable
to the search for similarities in the amino acid sequence of two proteins.
J. Mol. Biol. *48* , 443–453. [https://doi.org/10.1016/0022-2836(70)90057-4](https://doi.org/10.1016/0022-2836(70)90057-4)[.](https://doi.org/10.1016/0022-2836(70)90057-4)

6 5. [Delano, W. (2002). Pymol: An open-source molecular graphics tool. CCP4](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref65)
[Newsletter Protein Crystallography,](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref65)[](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref65)[82–92](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref65)[.](http://refhub.elsevier.com/S2666-3899(25)00267-3/sref65)

6 6. Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac,
P., Rault, T., Louf, R., Funtowicz, M., et al. (2019). HuggingFace’s
Transformers: State-of-the-art Natural Language Processing. Preprint at
arXiv. [https://doi.org/10.48550/arXiv.1910.03771](https://doi.org/10.48550/arXiv.1910.03771)[.](https://doi.org/10.48550/arXiv.1910.03771)

6 7. Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M.,
Zettlemoyer, L., and Stoyanov, V. (2019). RoBERTa: A Robustly Optimized
BERT Pretraining Approach. Preprint at arXiv. [https://doi.org/10.48550/](https://doi.org/10.48550/arXiv.1907.11692)
[arXiv.1907.11692](https://doi.org/10.48550/arXiv.1907.11692)[.](https://doi.org/10.48550/arXiv.1907.11692)

6 8. Xie, X. jbloomlab SARS2_RBD_Ab_escape_data Xie_XS. [https://media.](https://media.githubusercontent.com/media/jbloomlab/SARS2_RBD_Ab_escape_maps/refs/heads/main/processed_data/escape_data.csv)
[githubusercontent.com/media/jbloomlab/SARS2_RBD_Ab_escape_maps/](https://media.githubusercontent.com/media/jbloomlab/SARS2_RBD_Ab_escape_maps/refs/heads/main/processed_data/escape_data.csv)
[refs/heads/main/processed_data/escape_data.csv](https://media.githubusercontent.com/media/jbloomlab/SARS2_RBD_Ab_escape_maps/refs/heads/main/processed_data/escape_data.csv)[.](https://media.githubusercontent.com/media/jbloomlab/SARS2_RBD_Ab_escape_maps/refs/heads/main/processed_data/escape_data.csv)

6 9. Greaney, A.J., Starr, T.N., and Bloom, J.D. (2022). An antibody-escape
estimator for mutations to the SARS-CoV-2 receptor-binding domain.
Virus Evol. *8* , veac021. [https://doi.org/10.1093/ve/veac021](https://doi.org/10.1093/ve/veac021)[.](https://doi.org/10.1093/ve/veac021)

7 0. Virtanen, P., Gommers, R., Oliphant, T.E., Haberland, M., Reddy, T.,
Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J.,
et al. (2020). SciPy 1 .0: fundamental algorithms for scientific computing
in Python. Nat. Methods *17* , 261–272. [https://doi.org/10.1038/s41592-](https://doi.org/10.1038/s41592-019-0686-2)
[019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)[.](https://doi.org/10.1038/s41592-019-0686-2)

7 1. Wu, X., Yang, Z.-Y., Li, Y., Hogerkorp, C.-M., Schief, W.R., Seaman, M.S.,
Zhou, T., Schmidt, S.D., Wu, L., Xu, L., et al. (2010). Rational Design of
Envelope Identifies Broadly Neutralizing Human Monoclonal Antibodies
to HIV-1. Science *329* , 856–861. [https://doi.org/10.1126/science.](https://doi.org/10.1126/science.1187659)
[1187659](https://doi.org/10.1126/science.1187659)[.](https://doi.org/10.1126/science.1187659)

7 2. Alamyar, E., Duroux, P., Lefranc, M.-P., and Giudicelli, V. (2012). IMGT(®)
tools for the nucleotide analysis of immunoglobulin (IG) and T cell receptor
(TR) V-(D)-J repertoires, polymorphisms, and IG mutations: IMGT/V-
QUEST and IMGT/HighV-QUEST for NGS. Methods Mol. Biol. *882* ,
569–604. [https://doi.org/10.1007/978-1-61779-842-9_32](https://doi.org/10.1007/978-1-61779-842-9_32)[.](https://doi.org/10.1007/978-1-61779-842-9_32)

7 3. Alamyar, E., Giudicelli, V., Li, S., Duroux, P., and Lefranc, M.P. (2012). IMGT/
HighV-QUEST: The IMGT web portal for immunoglobulin (IG) or antibody
and T cell receptor (TR) analysis from NGS high throughput and deep
sequencing. Immunome Res. *8* , 26–40. [https://doi.org/10.4172/1745-](https://doi.org/10.4172/1745-7580.1000056)
[7 580.1000056](https://doi.org/10.4172/1745-7580.1000056)[.](https://doi.org/10.4172/1745-7580.1000056)

7 4. Li, S., Lefranc, M.-P., Miles, J.J., Alamyar, E., Giudicelli, V., Duroux, P.,
Freeman, J.D., Corbin, V.D.A., Scheerlinck, J.-P., Frohman, M.A., et al.
(2013). IMGT/HighV QUEST paradigm for T cell receptor IMGT clonotype
diversity and next generation repertoire immunoprofiling. Nat. Commun. *4* ,
2 333. [https://doi.org/10.1038/ncomms3333](https://doi.org/10.1038/ncomms3333)[.](https://doi.org/10.1038/ncomms3333)

7 5. Giudicelli, V., Duroux, P., Lavoie, A., Aouinti, S., Lefranc, M.-P., and
Kossida, S. (2015). From IMGT-ONTOLOGY to IMGT/HighVQUEST
for NGS Immunoglobulin (IG) and T cell Receptor (TR) Repertoires in
Autoimmune and Infectious Diseases. Autoimmun Infec Dis *1* . [https://](https://doi.org/10.16966/2470-1025.103)
[doi.org/10.16966/2470-1025.103](https://doi.org/10.16966/2470-1025.103)[.](https://doi.org/10.16966/2470-1025.103)

7 6. Stewart-Jones, G.B.E., Chuang, G.-Y., Xu, K., Zhou, T., Acharya, P.,
Tsybovsky, Y., Ou, L., Zhang, B., Fernandez-Rodriguez, B., Gilardi, V.,
et al. (2018). Structure-based design of a quadrivalent fusion glycoprotein
vaccine for human parainfluenza virus types 1–4. Proc. Natl. Acad. Sci.
USA *115* , 12265–12270. [https://doi.org/10.1073/pnas.1811980115](https://doi.org/10.1073/pnas.1811980115)[.](https://doi.org/10.1073/pnas.1811980115)

7 7. Georgiev, I.S., Joyce, M.G., Yang, Y., Sastry, M., Zhang, B., Baxa, U.,
Chen, R.E., Druz, A., Lees, C.R., Narpala, S., et al. (2015). Single-Chain
Soluble BG505.SOSIP gp140 Trimers as Structural and Antigenic
Mimics of Mature Closed HIV-1 Env. J. Virol. *89* , 5318–5329. [https://doi.](https://doi.org/10.1128/JVI.03451-14)
[org/10.1128/JVI.03451-14](https://doi.org/10.1128/JVI.03451-14)[.](https://doi.org/10.1128/JVI.03451-14)

7 8. Holt, C.M., Figshare, [https://doi.org/10.6084/m9.figshare.29647952](https://doi.org/10.6084/m9.figshare.29647952)[.](https://doi.org/10.6084/m9.figshare.29647952)

<||WXb23TXrUn3Rxz00yNNr89HV||>
