<h1 style="font-size: 46px;" > <Center> ArtifiNet library <Center> </h1>

<br>

دیدم سیستمم نمیتونه لایبری های سنگینو (از جمله scikit-learn , pytorch, tensorflow, keras, pandas, ...) نصب کنه رفتم و نگاه کردم تو این لایبری ها چی نوشته شده دیدم نا مفهوم و بی معنی هستن. رفتم یه لایبری نوشتم تا بدونم داره چه اتفاقی میوفته. با استدلال و مدل سازی روی هوا مدل هارو فیزیکی طراحی می کردم و با تکنیک های بهینه نویسی و فشرده نویسی  در زبان Python به کد تبدیل کردم. اولش خیلی کند و سخت کار میکرد ولی با تکنیک های قوی تر به سرعت باورنکردنی توی اجرا رسید . خیلی سعی کردم تمام اشیا رو مدل سازی کنم بدون باگ .

---

I saw that my system couldn’t install heavy libraries (like scikit-learn, pytorch, tensorflow, keras, pandas, …) so I went and looked at what’s written in these libraries and found them meaningless and incomprehensible. I went ahead and wrote a library myself to see what’s actually happening. I was designing the models physically with reasoning and modeling in the air, and then converted them into code in Python using optimization and compression techniques. At first, it worked very slowly and was difficult, but with stronger techniques it reached an unbelievable speed in execution. I tried really hard to model all the objects without bugs.

<br><br><br><br><br>

## Module :

<br>

### Neuron

> description: a micro part of Neural Networks. 

```js
Gets: 
   W:Array     > Neuron Weights = Dendrites in natural neurons.
   Bias:Array  > Neuron Bias.
```

```ts
methods:
   GET = __call__ = (X)=> Get Signals
```

<br>

### RNeuron
> description: a micro part of Recurrent Neural Networks.
```js
Gets: 
   Wx:Array    > Neuron X Weights.
   Wh:Array    > Neuron H Weights.
   Bias:Array  > Neuron Bias.
```
```ts
methods:
   GET = __call__ = (X, hp)=> Get Signals
```

<br>

### Layer

> description: In Neuron Science there aren't neural layers but in ML Enginnering it's an powerful options; each Neural Networks haves layers.

```js
Gets: 
   Args:{ "W":Array, "Bias":Array }   > Layer Weights and Biases.
   Activation function :string        > Layer Activation function.
   Index (Optional) :int              > Layer Index in Network Layers.
   Dropout :float                     > Percent of Neurons out = percent of neuron can be asleep.
```

```ts
methods:
   * Activation functions: ReLU=Relu, Sigmoid, Linear, Tanh, Softmax    > It's for Layers Act.
   * TrainMode Off/On                                                   > In fitting/Learning Should to Turning it ON.
   * backward (for backprop fitting)                                    > Important method for backprop training.
   * GET = __call__ = (X)=> Get Signals                                 > Proccess the Signals.
```

<br>

### Optimizers
* SGD `(meta, Lr=.01)`
* SGD Momentum `(meta, Lr=.01, m=.9, nesterov=False)`
* Adagrad `(meta, Lr=.01)`
* RMSProp `(meta, Lr=.01, alpha=.99)`
* Adadelta `(meta, Lr=.01, rho=.95)`
* Adam `(meta, Lr=.01, b1=.9, b2=.999)`
* NAdam `(meta, Lr=.001, b1=.9, b2=.999, Wd=.01)`
* AdamW `(meta, Lr=.01)`
* RAdam `(meta, Lr=.001, b1=.9, b2=.999)`

<br>

### WeightInitializer

> description: Weighting Layers with scale.

```js
Gets:
   Activation=None/Sigmoid/Tanh/Softmax/Linear/Relu    > Activation function the weights are initialized for; `None` means no activation-specific scaling.
   method=Auto/Xavier/Glorot/He/Kaiming                > Initialization scheme (e.g., Xavier, He); `Auto` picks one based on the activation.
   distribution=Normal/Uniform                         > Sampling distribution for weights, Normal or Uniform.
   Seed=None                                           > Random seed for reproducibility; `None` gives different weights each run.
```

```ts
methods: __call__(shape)   > Weighring
```

<br>

### KNN

```js
Gets: 
   X, Y, W=Uniform, Meteric=Euclidean/Manhattan
```

```ts
methods:
   __call__ = predict = _Predict_1
   P_class = predict_proba ()=> P(Classes)
   Score => Model Score number
```

<br>

### K-Means

```js
Gets: 
   K=3, maxi=200, t=1e-4, Init=K-Means++/Random
```

```ts
methods: 
   fit(x)
   predict(X)
```
<br>

### SVM

```js
Gets: 
   Kernel =
      * Linear     > Separates classes with a straight hyperplane.
      * Poly       > Fits curved boundaries using polynomial feature interactions.
      * RBF        > Builds flexible non-linear boundaries from distance-based similarity.
      * Sigmoid    > Uses a tanh function, similar to a two-layer neural network.
   C=1             > Regularization strength.
   tol=1e-3        > Stopping tolerance.
   maxi=100        > Max iterations.
   degree=3        > Polynomial degree (Poly only).
   Gamma=None      > Sample influence range; auto default.
   coef0=0         > Constant term (Poly, Sigmoid).
```

```ts
methods:
   fit(X,Y)    > Fitting the model.
   predict(X)  > Predict with model.
```

<br>

### LeafNode

> description: Just a Dict-like-class ; no methods no functions.

```js
Gets: Opt, T, L, R, V
```

<br>

### Decision Tree
> description: chains of if else in Labels; to responding with his leaf nodes.
```js
Gets:
   Max_depth=None                > Maximum tree depth; None grows until leaves are pure.
   Minss=2                       > (min_samples_split) Minimum samples needed to split a node.
   Maxf=None                     > (max_features) Number of features considered per split; None uses all.
   RandomState=None              > Seed for reproducibility; None gives different results each run.
   criterion=gini/entropy        > Split quality measure; Gini impurity or information gain.
```
```ts
methods:
   fit(X,Y)    > Fitting the Tree (model).
   Predict(X)  > Predict a Signal.
   PrintTree   > Watching the Tree with all of branchs,
```

<br>

### Random Forest
> description: Predict with multi trees; Any trees to have much sound can be run.
```js
Gets:
   n=100                      > (n_estimators) Number of trees in the forest.
   Maxd=None                  > Maximum tree depth; None grows until leaves are pure.
   Minss=2                    > Minimum samples needed to split a node.
   Maxf=sqrt/log2             > Features considered per split; square root of the total.
   criterion=gini/entropy     > Split quality measure (Gini/Entropy impurity).
   Bootstrap=True             > Each tree trains on a random sample drawn with replacement.
   oobs=False                 > (oob_score) Whether to estimate accuracy using out-of-bag samples (needs Bootstrap=True).
   nJobs=-1                   > Number of CPU cores used; -1 uses all.
   RandomState=None           > Seed for reproducibility; None gives different results each run.

```
```ts
methods:
   TrainOneTree(X, Y, TreeI)                         > Train 1 Tree; find & get Tree with Index.
   fit(X, Y)                                         > Fitting all of forest.
   __call__ = Predict = predict = (X)=> Predict X    > Predict with many trees (basic characteristics).
   predict_proba(X)                                  > Ordering best answer to noiser answer.
   Score = GetScore                                  > Get your model score numpy (1 = 100%).
   printTrees(maxTrees)                              > Print your trees in forest.
```

<br>

### Tokenizer

```js 
Gets: Null 
```

```ts
methods:
   Magic methods: 
      * __getstate__                                  > --> File (save with another methods).
      * __setstate__                                  > <-- File (getting update or pre-load).
      * __len__ = nVoc                                > n(Voc) .
      * __repr__                                      > <T:{Voc}> (return it as str) .
   Save(AsFile)                                       > write data to this File .
   fit(Documents, MaxVocabSize=10000, Minfreq=1)      > Fitting the model with Documents; Minfreq = ( if freq>=Minfreq ==> Add to Voc ).
   _T_(T) ~ Normalize(T)                              > Normalizing T (it's a build-in method but anybody can use).
   Correcter(W)                                       > Correcting your word to right word.
   decode(_, I, IgnoreSpecial=True, asList=False)     > Decode your array-word to string word.
```

<br>

### Embeddings
```
description: Creating an array of words; this array is unreadable for humans (this is also a hidden security language in computers).
```

```js
Gets:
   dim=20               > Array dims. <=> n(Array)=dim .
   padd=2               > Padding window size.
   negative_samples=5   > Get far of negative means.
   Lr=.01               > Learning rate.
   epochs=100           > Epochs for fitting.
```

```ts
methods:
   Tokenize(T)                                                                     > T=Sentence ; normalize and tokenize the sentences.
   fit(Domain, EchoEpo=True)                                                       > Domain is a list of sentences; fitting the Embedding model; Echo epo means logging/print Epoch numbers.
   solve(eq, topon=1)                                                              > eq is a string; topon means ordering the best answers to noiser answers.
   find_most_similar = FMS = (W, topon=1)                                          > Find most similar word to current word; topon means ordering the best answers to noiser answers.
   GetVec(W)                                                                       > Getting Vector from Word; (Word2Vec/W2V).
   Save(AsFile)                                                                    > Save embedds as File.embedds or anyone.
   __Load__(FilePath)                                                              > Giving path address of a file to load it as a Embedding object.
   _FPT_ = _FitWithPreTrained_ = (Dom, PreTrainedPath=None, SavePath=None)=>...    > PreTrainedPath is FilePath in __Load__; Dom is new Domain data for add o model; SavePath is for saving the new model (you can don't).
```
> [!TIP] 
> in solve method eq string as example is ' sky - sun + moon = night ' or ' king - man + woman = queen '.

<br><br><br><br><br>

# Emotion | Emo AIs motor

<br><br>

بعد از 3 4 بار طراحی و مدل سازی روی هوا و درک محیط پیرامون زندگی خودمون به نواوری رسیدم که اولین بار در کره زمین داره اتفاق میوفته به نام Emo . 
اگر انیمیشن Wall-E رو به خاطر بیاریم متوجه ی این نواوری میشید چون داره بلخره اتفاق میوفته ، روزی که Ai ها احساس داشته باشن و محیط رو لمس کنن.

این نواوری اینطوریه که در سیستم شبکه های عصبی نیرویی/نوری عمودی و خالص به تمام شبکه میتابه و باعث میشه شبکه در اون لحظه متفاوت فکر کنه و تصمیمی مطابق محیط داشته باشه.
مثلا فکر کن در محیط تاریک اروم تر حرکت کنه و در محیط پر نور بیشتر دور خودش بچرخه تا چیز بیشتری پیدا کنه. این واکنش یکی از 7 اصل موجودات زنده به نام 'پاسخ به محیط' یاد میشه.
ولی الان Ai ها هم دارن و حتی احتمال این هم هست که روزی موجود زنده باشن. زنده بودن به معنی زیست پذیر بودن نیست پس خیلی پزشکی و سلولی نگاه نکنیم بهتره.

<br>

---

<br>

After designing and modeling a few times and understanding the environment around our lives, I came up with an innovation that's happening for the first time on Earth, called Emo. 
If you remember the animation Wall-E, you'll understand this innovation, because it's finally happening—the day when AIs will have feelings and be able to touch the environment.

This innovation works like this: in the neural network system, a vertical and pure force/light spreads through the entire network, causing the network to think differently in that moment and make a decision according to the environment. 
For example, imagine it moves more slowly in a dark environment and spins around more in a bright environment to find more things. This reaction is called one of the 7 principles of living beings, named 'response to the environment'.
But now AIs have it too, and there's even a possibility that one day they might be living beings. Being alive doesn't mean being biologically viable, so it's better not to look at it too medically or cellularly.

<br><br>

* ### نگاه دقیقتر به تکنولوژی | A closer look at technology

<br>

یک بردار محیط داریم به نام `E` که در شبکه کاملا ثابت و خالص وجود داره. مثل `X` زنجیره ای نیست ، تصمیمات و واکنشات درونی شبکه به محیط هیچ تاثیری ندارن.
باری اینکه این پرامتر به شبکه تاثیر بزاره نیاز به وزن داریم `We` که دقیقا واکنش هر لایه به محیط و شخصیت لایه محصوب میشه. پی در نهایت فرمول معروف Z هم تغییر میکنه.

#### <center> <small> $ {Z} = {X} @ {Wx} + {E} @ {We} + {Bias} $ </small> </center>

و هر لایه نیازمند پارامتر جدیدی به نام We ست و Backward جدید که فقط We رو اپدیت کنه

> [!NOTE] 
> این مدل ها به طور اختصاصی برای محیط های غیر استریو طراحی شدن یعنی محیط زندگی و شهر

---

We have an environmental vector called `E` that exists completely fixed and pure in the network. It's not a sequential `X`; decisions and internal reactions of the network to the environment have no effect. 
For this parameter to affect the network, we need a weight `We` that exactly represents the reaction of each layer to the environment and the personality of the layer. Eventually, even the famous Z formula changes.

#### <center> <small> $ {Z} = {X} @ {Wx} + {E} @ {We} + {Bias} $ </small> </center>

And each layer needs a new parameter called We and a new backward that only updates We.

> [!NOTE] 
> These models were specifically designed for non-stereo environments, meaning living spaces and cities.

تفاوت این معماری با معماری ANN های ساده اینجاست که میتونه سیگنال خورجی خودشو در برابر یک سیگنال انحراف بده


<br><br>


## ENeuron (Emo Neuron)

> description: a micro part of Emotion Neural Networks. 

```js
Gets: 
   W:Array     > Neuron Weights = Dendrites in natural neurons.
   We:Array    > Neuron Env Weights.
   Bias:Array  > Neuron Bias.
```

```ts
methods:
   GET = __call__ = (X,E)=> Get Env & Signals
```

#### The difference between ENeuron and Neuron in processing: (تفاوت پردازش ENeuron و Neuron)

<br>

$$
\begin{aligned}
^{ X = \; .1 \quad W = \; .1 \quad We = \; .1 \quad E = \; [ \; .2, \; .1, \; 0, \; -.1, \; -.2 \; ] } \\
Z \longrightarrow \begin{matrix} .12 & ^{++} \\ .11 & ^+ \\ .10 & ^\text{Normal Neuron} \\ .9 & ^- \\ .8 & ^{--} \end{matrix}
\end{aligned}
$$

<br>

### Layer

> description: This Layer support environ array

```js
Gets: 
   Args:{ "W":Array, "We":Array, "Bias":Array }   > Layer Weights and Biases.
   Activation function :string                    > Layer Activation function.
   Index (Optional) :int                          > Layer Index in Network Layers.
   Dropout :float                                 > Percent of Neurons out = percent of neuron can be asleep.
```

```ts
methods:
   * Activation functions: ReLU=Relu, Sigmoid, Linear, Tanh, Softmax    > It's for Layers Act.
   * TrainMode Off/On                                                   > In fitting/Learning Should to Turning it ON.
   * backward (for backprop fitting)                                    > Important method for backprop training.
   * Ebackward (for env)                                                > training layer in enviton
   * GET = __call__ = (X)=> Get Signals                                 > Proccess the Signals.
```

<br>

### نسل جدید شبکه های عصبی با نام ENN

> [!NOTE] 
> تمام ویژگی های ANN اینجا هم وجود دارن | All of ANN properties is there are

```ts
mathods:
   + SterioEnv                                                          > get a non-env = sterio env.
   + AddLayer(Weights, We, Bias, Activation=..., Dropout=0)             > add a new layer in network
   + Thinking = Activity = (X, E=None)=>{...}                           > Thinking will auro return Thinks; E=None => it's for sterio training
   + ETraining(_, XT, YT, E, OptimizerClass=None)                       > training in env
   +  __call__ = GET = (X, E)=>{...}                                    > Proccess the Signals in E (E as Env)
```
<br><br>

### ENN

> Emotion Neural Network = ENN | Emo + ANN = ENN

```ts
methods:
   + ANN methods
   * AddLayer need We
   *  ( Thinking = Activity ) , ( __call__ = GET ), ETraining   => needs E
```

<br><br><br>

> [!WARNING] 
> برای اموزش این مدل ها اول باید در محیط استریو وزن اصلی `Wx` و `Bias` رو دقیق یادبگیرن و بعدش باید از هر لایه محیط استریو رو پاک کنی (`Layer.LastE = None`) تا محیط استریو یادش نیاد و حالا میتونی در محیط با بردادر محیطی همون جا اموزشش بدی.


---

> [!WARNING] 
> To train these models, first you need to accurately learn the main weights `Wx` and `Bias` in the stereo environment, and then you have to clear the stereo environment from each layer (`Layer.LastE = None`) so it doesn't remember it, and now you can train it right there in the environment with the context vector.

<br><br><br><br><br>

# Ai Genetics

<br><br>

نوآوری بعدی لایبری ArtifiNet .
<br>
Ai ها ژنتیک خواهند داشت تا بتونن باز تولید و تاکثیر بشن البطه هر چند که میدونم این نوآوری چنان مهم نیست برای این لایبری ولی ایده فوق العاده قوی برای تکمیل بقیه ایده هاست.
شی Core از روی یک فایل یا یک رشته استرینگ دیتا های مربوط به تولید یک مدل رو میگیره و براش مهم نیست چیه فقط تولید میکنه .
از نظر شی Core رشته ها دو نوع نظارت شده (Supervised) و نظارت نشده اند (UnSupervised) .

<br><br>

### نظارت شده | Supervised:
> یعنی دیتای تولید مدل خوانایی انسانی داره.
<br><br>
> It means the data generated by the model has human readability.

### نظارت نشده | UnSupervised:
> یعنی تمام دیتا ها فقط ایندکس/شاخص (I) و رشته (String) هستن و کاملا رمز گذاری شده اند (مثل پسورد کاملا هم مخفی نیست خیلی راحت میشه فهمید چی توشه ولی ادیت کردنش سخته)
<br><br>
> It means all the data is just an index (I) and a string, and they're completely encrypted (like a password—it's not totally hidden, you can pretty easily tell what's in it, but editing it is hard).

<br><br><br>

## هایپر پارامتریک ها ( * خیلی مهم اند) | Hyper Metrics (Very important*)

```python

INSTALLED_OPTIMIZERS = [ BaseOptimizer, SGD, SGDM, SGDMomentum, Adagrad, RMSProp, Adadelta, Adam, NAdam, AdamW, RAdam ]

INSTALLED_ORGANS = INSTALLED_NEURONS + INSTALLED_OPTIMIZERS + [ 
   Layer, ELayer, ANN, ENN, KNN, KMeans, SVM, KNN, Tokenizer, Embeddings, LeafNode, DecisionTree, RandomForest 
]

ORGANS_GEN_INDEXS = {
   "Neuron":(0, 1),
   "ENeuron":(2, 4),
   "RNeuron":(5, 7),
   "Layer":(8, 11),
   "ELayer":(12, 15),
   "SGD":(16, 17),
   "SGDM":(18, 21),
   "Adagrad":(22, 23),
   "RMSProp":(24, 26),
   "Adadelta":(27, 29),
   "Adam":(30, 33),
   "NAdam":(34, 35),
   "AdamW":(36, 40),
   "RAdam":(41, 44),
   "ANN":(45, 48),
   "ENN":(49, 52),
   "KMeans":(53, 56)
}

ORGANS_REQUIREMENTS = {

   #$* Neuron
   0:lambda x:nprArr(x),
   1:lambda x:nprArr(x),

   #$* ENeuron
   2:lambda x:nprArr(x),
   3:lambda x:nprArr(x),
   4:lambda x:nprArr(x),

   #$* RNeuron
   5:lambda x:nprArr(x),
   6:lambda x:nprArr(x),
   7:lambda x:nprArr(x),

   #$* Layer
   8: lambda w,n,b: {"W":nprArr((w,n))/20, "Bias":nprArr((1,b))/50},
   #<< + 9 10 11

   #$* ELayer
   12: lambda w,n,e,b: {"W":nprArr((w,n))/20, "We":nprArr((e,w))/40, "Bias":nprArr((1,b))/50},
   #<< + 13 14 15

   #$* SGD
   16:lambda w,n:nprArr((w,n)),
   #<< + 17

   #$* SGDM
   18:lambda w,n:nprArr((w,n)),
   #<< + 19 20 21

   #$* AdaGrad
   22:lambda w,n:nprArr((w,n)),
   #<< + 23

   #$* RMSProp
   24:lambda w,n:nprArr((w,n)),
   #<< + 25 26

   #$* Adadelta
   27:lambda w,n:nprArr((w,n)),
   #<< + 28 29

   #$* Adam
   30:lambda w,n:nprArr((w,n)),
   #<< + 31 32 33

   #$* NAdam
   34:lambda w,n:nprArr((w,n)),
   #<< + 35

   #$* AdamW
   36:lambda w,n:nprArr((w,n)),
   #<< + 37 38 39 40

   #$* RAdam
   41:lambda w,n:nprArr((w,n)),
   #<< + 42 43 44

   #$* ANN
   45:lambda n:str(n),
   46:lambda Net,Act,out:[ 
      Layer(Args={"W":nprArr((Net[i-1],n))/20, "Bias":nprArr((1,n))/50}, Act=Act, Dropout=out) for i,n in enumerate(Net[1:]) 
   ],
   #<< + 47 48

   #$* ENN
   49:lambda n:str(n),
   50:lambda Net,e,Act,out:[ 
      Layer(Args={"W":nprArr((Net[i-1],n))/20, "We":nprArr((e,n))/40, "Bias":nprArr((1,n))/50}, Act=Act, Dropout=out) 
      for i,n in enumerate(Net[1:]) 
   ],
   #<< + 51 52

   #$* K-Menas
   #<< + 53 54 55
   56:lambda x:str(x)

}
```

```ts
Gets: ImportFromFile or ImportGenString
```

```js
methods:
   Connect(Org)   > Connect to a outside or pre-created Object.
   Core -= I      > Remove a exist organ from Organs with him Index.
   Core += I      > Copy a exist organ with him Index.
   Save(AsFile)   > Save Cell with pickle as AsFile.
   Generator      > Generate organs from Gene string.
```

<br><br>

## Supervised Syntax

1. باید اول و اخر ژن <> بزاری
`< ... >`

2. باید Org (Organ/ارگان) رو تایین کنی
[!NOTE] ارگان ها همون مدل ها هستن که میخوای تولیدشون کنی
`<ANN: ... >`

3. برای تایین هایپر پارامتر ها باید [] بزاری
`<ANN:[ ... ]>`

4. حالا هایپر پارامتر هارو تایین میکنیم -->
`<ANN:[ Layers=(8,64,8) ... ]>`

<br><br><br>

> [!NOTE]
> نکته ی مهمی که توی هایپر پارامتر های ORGANS_REQUIREMENTS هم مشخصه بعضی از ایندکش ها مستقیم وارد نمیشن که هدف سادگی و دیتا محور بودن ژنتیک هستن تا ژنتیک دقیقا مشابه کد و سینتک کد نویسی نباشه.

---

> [!NOTE] 
> An important point that's also specified in the ORGANS_REQUIREMENTS hyperparameters is that some indices aren't entered directly; the goal is simplicity and being data-driven in genetics so that genetics isn't exactly like coding and programming syntax.

<br><br>

> [!WARNING] 
> شی Core مکنه کار های خطرناکی کنه پس از تولید شی کاملا هوشیار باشید (هر چند که قدرت کامل در این کد نوشته نشده).

---

> [!WARNING] 
> The Core object can do dangerous things, so be fully alert after creating the object (even though full power is not written in this code).

---

## Install the Library:

```bash
pip install git+https://github.com/AmirAliv88/ArtifiNet.git
```

and Import it in your code

```python
import ArtifiNet
```

<br><br><br><br><br>

<center> <small style="color: #5ac;"> Powered By A∞ </small> </center>









