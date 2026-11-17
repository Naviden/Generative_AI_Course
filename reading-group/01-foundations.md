# 01 — Shared foundations for generative AI

[Download as PDF](https://github.com/Naviden/Generative_AI_Course/raw/refs/heads/main/reading-group/pdf/01-foundations.pdf)

[Reading-group home](README.md) · Next: [02 — How generative models work](02-generative-models.md)

## 1. What a generative model does

Suppose we want to generate new records describing objects by their colour and size. One record might say “blue and large”; another might say “not blue and not large”. If 40 out of 100 real records are blue, a generator that makes every record blue gets the frequencies wrong, even though each individual record is possible. We want to learn **which combinations occur and how often to generate each one**.

A **distribution** specifies the possible outcomes and their probabilities. In our example, it is a table saying how likely each colour–size combination is. A **generative model** uses such a distribution to produce new observations. An **observation**, written $`x`$, is one complete item: here, a record such as “blue and large”. A **variable** is one attribute of that item: colour can be blue or not blue; size can be large or not large. These categories describe all the information our simple model stores. It does not generate an object's shape, an image or an individual identity.

For an image generator, the observation would instead be a whole image. Both tasks require more than producing one acceptable output: the model must decide how often to produce different outputs. We will use the small record table because we can inspect every probability and calculation directly.

### From observed records to a model

Our **dataset** is a collection of 100 observed records: 40 are blue, 30 are large and 20 are both. Every record belongs to exactly one of the following four rows. Of the 40 blue records, 20 are also large, leaving `40 − 20 = 20` blue but not large. Of the 30 large records, 20 are blue, leaving `30 − 20 = 10` large but not blue. The remaining `100 − 20 − 20 − 10 = 50` are neither.

| Complete observation | Count in the dataset | Probability in a frequency-fitted model |
|---|---|---|
| Blue and large | 20 | 0.20 |
| Blue and not large | 20 | 0.20 |
| Not blue and large | 10 | 0.10 |
| Not blue and not large | 50 | 0.50 |

To obtain the last column, divide each count by 100: for example, `20/100 = 0.20`, or 20%. **Fitting** this table means setting its probabilities from the observed counts. The four probabilities add to `0.20 + 0.20 + 0.10 + 0.50 = 1`, accounting for all possible generated records. A probability of zero would mean a row is never generated; a probability of one would mean it is always generated.

The adjustable probabilities are the model's **parameters**, collectively written $`\theta`$. Here, $`\theta`$ is simply the list `(0.20, 0.20, 0.10, 0.50)`. The notation $`p_\theta(x)`$ means “the probability this parameter choice assigns to observation $`x`$”. For example, $`p_\theta(\text{blue and large})=0.20`$. If we change the list to `(0.10, 0.20, 0.10, 0.60)`, blue-and-large records become less likely and neither-blue-nor-large records become more likely. We moved probability between rows so that the total remained one.

The **population** is the wider source of records we want the generator to represent. For example, the 100 records might describe objects inspected from a factory's output; the population includes the other objects from that same production process. Another batch of 100 might contain 38 blue objects rather than 40. So a frequency of 40% in our dataset is an estimate, not proof that exactly 40% of the whole population is blue. The population's unknown distribution is written $`p_{\mathrm{data}}`$; our fitted approximation is $`p_\theta`$. Later we will check the approximation on records not used to fit it.

### What it means to generate from the model

**Sampling** means randomly choosing an outcome using the model's probabilities. Imagine a spinner divided into four sectors covering 20%, 20%, 10% and 50% of its circle. Label the sectors with the four rows and spin it: the larger sectors are more likely to be selected, but the same sector need not win every time.

A computer can do the equivalent by drawing a number uniformly from zero up to, but not including, one. “Uniformly” means that equal-length intervals have equal chances: falling between 0.20 and 0.30 is as likely as falling between 0.70 and 0.80. Use the following rule:

| Random number | Generated record |
|---|---|
| From 0 up to, but not including, 0.20 | Blue and large |
| From 0.20 up to, but not including, 0.40 | Blue and not large |
| From 0.40 up to, but not including, 0.50 | Not blue and large |
| From 0.50 up to, but not including, 1 | Not blue and not large |

A draw of 0.34 selects the second row. The notation $`X\sim p_\theta`$ means “draw the observation $`X`$ from the model in this way”. Capital $`X`$ refers to the outcome before the draw; lowercase $`x`$ can denote a particular result, such as “blue and not large”.

In 10 draws, we expect two blue-and-large records on average, but a particular batch could contain one, three or none. A 20% probability is not a promise of exactly two successes in every batch of 10. Each draw uses the same probabilities, regardless of the previous result; over many such draws, the proportion tends towards 20%. Always choosing the most probable row would instead produce “not blue and not large” every time: 100% of the outputs would be in a row assigned only 50% probability. That is a different generation rule from sampling our table.

**Exercise:** use the interval table above. What does a random draw of 0.47 generate, and does drawing again change the model?

**Worked answer:** 0.47 lies in the third interval, so the generated record is “not blue and large”. Drawing again can change the output, but the four probabilities remain fixed. **Sampling changes the observation; learning changes the distribution used to sample it.**

## 2. Joint and conditional probability

Our table is a **joint distribution** because it assigns probabilities to colour and size together: “blue and large” has probability 0.20. If we only ask about colour, add the two blue rows: `0.20 + 0.20 = 0.40`. If we only ask about size, add the two large rows: `0.20 + 0.10 = 0.30`.

An **unconditional** request is “generate a record”: all four rows are available. A **conditional** request is “generate a record that is large”. Now only two rows are allowed: the 20 blue-and-large records and the 10 not-blue-and-large records. The relevant group has 30 records, not 100. The blue share within this smaller group is `20/30`, not `40/100`.

An **event** is a condition an outcome can satisfy. Let $`A`$ mean “the record is blue” and $`B`$ mean “the record is large”. Then $`p(A,B)`$ means the probability of both: `20/100 = 0.20`. The vertical bar in $`p(A\mid B)`$ is read “given”: it asks for the probability of blue **given that we know the record is large**.

The general rule is $`p(A\mid B)=p(A,B)/p(B)`$. Here it gives `0.20/0.30 = 2/3`, the same result as `20/30`. Dividing by $`p(B)`$ makes the probabilities within the allowed group add to one: `2/3` blue plus `1/3` not blue. This division requires $`p(B)>0`$; if our model assigned zero probability to large records, the table alone could not tell us how to sample within that group.

**Exercise:** among 100 records, 40 are blue, 30 are large and 20 are both. What is the probability of blue given large, and is it the same as large given blue?

![Among 100 records: 20 are both blue and large, 20 blue only, 10 large only, and 50 neither. Conditioning on large gives 20/30; conditioning on blue gives 20/40.](images/conditional-probability.png)

**Worked answer:** “blue given large” starts with the 30 large records, of which 20 are blue: `20/30 = 2/3`. “Large given blue” starts with the 40 blue records, of which 20 are large: `20/40 = 1/2`. The numerator is the same overlap, but the denominator counts a different group. Among all records 40% are blue; among large records about 67% are blue. Knowing size therefore changes our prediction of colour in this table.

This does not show that making an object larger would turn it blue. For example, a factory could choose blue paint more often for its large product line. The table records which properties occur together; it contains no experiment in which size was changed while other factors were controlled.

For generation, a “large” request should therefore produce only large records, about two thirds blue and one third not blue. A “blue” request should produce only blue records, about half large and half not large. In text-to-image generation, a prompt such as “a blue car” similarly supplies information that should change which images are likely. The model can still generate different blue cars; conditioning does not specify a unique output. Unlike our exact table rule, a learned image model may fail to satisfy a prompt.

## 3. How a model learns from examples

The four probabilities in our table were fitted to observed frequencies. Why prefer them to 0.25 for every row? We need a score that compares models using the **same observed records**. **Likelihood** is the probability a model assigns to those observations, viewed as a function of its parameters: keep the records fixed and change the candidate probabilities.

For example, suppose two observed records, in order, are “blue and large” and “not blue and not large”. The equal-probability model assigns them probability `0.25 × 0.25 = 0.0625`; our fitted model assigns `0.20 × 0.50 = 0.10`. The fitted model gives this observed pair a larger likelihood. Multiplication assumes the records are **independent under the model**: knowing the first result does not change the probabilities for the second. Our generator makes separate draws from the same unchanged table. Records collected as dependent pairs would require a model of that dependence instead.

For the full dataset we multiply 100 factors: 20 factors of 0.20, 20 of 0.20, 10 of 0.10 and 50 of 0.50 for the fitted model. **Maximum-likelihood learning** chooses the permitted probabilities that make this product largest. We do not change which records were observed.

**Concrete example:** 50 of our 100 records are neither blue nor large. Suppose the model assigns that outcome only 5% probability. It repeatedly treats an ordinary observation as surprising: half the records receive a probability of just 0.05. This can happen before training, when the initial probabilities do not yet reflect the data; after inadequate training; or when a model learned from a different population where this outcome was rare. Maximum-likelihood learning adjusts the probabilities to make the observed data more probable. In our four-row table, fitting all 100 records gives this category 50% probability, while the other three categories share the remaining 50% according to their observed frequencies. Sampling from the fitted model will therefore produce this category about half the time, instead of about one time in twenty.

### Turning likelihood into a loss we can add

A **loss** is a numerical penalty; a smaller value is better according to that scoring rule. We use **negative log likelihood (NLL)**, also called **log-loss**, to turn the likelihood calculation into an additive penalty. For one record it is $`-\log p_\theta(x)`$: look up the probability assigned to the observed row, apply the natural logarithm (the `ln` key on a calculator), then change the sign.

What does `ln` calculate? It answers “to what power must we raise the number $`e\approx 2.71828`$ to obtain this value?” For example, $`e^{-0.693147\ldots}=0.5`$, so $`\ln(0.5)=-0.693147\ldots`$. You do not have to find that exponent by hand: the `ln` key calculates it. Throughout this lesson, $`\log`$ means this natural logarithm, not the base-10 logarithm found on many calculators' `log` key.

For probabilities between zero and one, `ln` is negative; the minus sign makes the penalty positive. A probability of one gives zero penalty. Smaller probabilities give larger penalties: the model is penalized more for treating an observation that actually occurred as unlikely.

| Probability assigned to the observed outcome | Negative log likelihood | Interpretation |
|---|---|---|
| 0.5 | $`-\ln(0.5)\approx-(-0.6931)=0.6931`$ | Smaller penalty |
| 0.1 | $`-\ln(0.1)\approx-(-2.3026)=2.3026`$ | Larger penalty |

To reproduce the first row, enter `0.5`, press `ln` to obtain about `−0.6931`, then change the sign to get `0.6931`. Repeat with `0.1` for the second row. These numbers come from the probability assigned to the record, not from a second learned parameter.

The useful logarithm rule is $`\log(ab)=\log a+\log b`$. In our two-record example, $`-\ln(0.20\times0.50)=-\ln(0.10)\approx2.3026`$. Calculating the penalties separately gives the same answer: $`-\ln(0.20)-\ln(0.50)\approx1.6094+0.6931\approx2.3026`$. We can therefore add penalties instead of multiplying many small probabilities. The logarithm preserves the ordering of positive numbers; changing its sign reverses that ordering. **The largest likelihood and the smallest NLL select the same model.** The displayed decimals are rounded; use the calculator's full values when adding them.

### Comparing models on all 100 records

To obtain **average NLL**, add one penalty for each observed record and divide by the number of records. An average makes the score per record: duplicating the dataset would double the total penalty but leave the average unchanged.

In the equal-probability model, every record gets probability 0.25, so every penalty is $`-\ln(0.25)\approx1.3863`$. Its average is therefore also 1.3863. For the fitted table, group records that have the same penalty:

| Observed group | Number of records | Penalty per record, rounded |
|---|---|---|
| Blue and large | 20 | $`-\ln(0.20)\approx1.6094`$ |
| Blue and not large | 20 | $`-\ln(0.20)\approx1.6094`$ |
| Not blue and large | 10 | $`-\ln(0.10)\approx2.3026`$ |
| Not blue and not large | 50 | $`-\ln(0.50)\approx0.6931`$ |

Using unrounded logarithms, the average is:

```math
\frac{20[-\log(0.20)]+20[-\log(0.20)]+10[-\log(0.10)]+50[-\log(0.50)]}{100}
\approx 1.2206.
```

Thus 1.2206 is better than 1.3863 **on these observations under this loss**. The fitted model does not improve the probability of every row: “not blue and large” drops from 0.25 to 0.10 and receives a larger penalty. But there are only 10 such records, while the 50 “neither” records benefit from a rise from 0.25 to 0.50. The total penalty falls. Raising every probability is impossible because their sum must remain one.

The change has a direct effect on generation. The equal-probability table produces a large record with probability `0.25 + 0.25 = 0.50`; the fitted table does so with probability `0.20 + 0.10 = 0.30`. Across many generated records, the large share should move from about 50% towards the observed 30%. **Learning changes the distribution; subsequent sampling uses that changed distribution.**

In this tiny model we can fit the probabilities directly by dividing counts by 100. A neural model instead stores adjustable numbers called **weights**, which determine output probabilities through a calculation. Training changes those weights to reduce a loss. The purpose is the same as changing our table: change which observations the generator is likely to produce. We do not need the mechanics of those updates here; they are in the optional further reading.

**Exercise:** half the training records are “not blue and not large”. Would assigning probability one to that row be an improvement over the fitted table?

**Worked answer:** assigning probability one to that row leaves zero for every other row. For each of the other 50 records, the model would say “this cannot happen”, although it did happen. One zero factor makes the full-data likelihood zero. The penalty `−ln(p)` grows without bound as `p` approaches zero, so an observed event assigned exactly zero probability has infinite NLL. Generating only the most common category is therefore a worse fit than representing all four observed categories.

This is one way to train a generator, not a rule that every generator must use. For example, a model can learn to remove deliberately added noise from an image and be scored on how accurately it predicts that noise. Lesson 2 examines that approach. Here, the important link is explicit: our score uses observed records to choose probabilities, and those probabilities govern the generated records.

## 4. How to judge a generator

Our fitted table scored 1.2206 on the records used to choose its probabilities, beating the equal-probability table's 1.3863. This tells us it fits **those records** better. It does not yet tell us whether it predicts new factory records better, or whether a program actually generates the right variety of outputs. We need to check both.

### Training fit and fresh observations

**Training data** are the observations used to fit the parameters. **Held-out data** are kept out of that fitting step. To evaluate on held-out records, freeze the probabilities, look up the probability of each newly observed record, and calculate average NLL exactly as before. Do not refit the table using those records before scoring it: that would answer how well the model fits data it has already seen.

For a worked example, suppose a separate batch of 100 records has counts `(18, 22, 12, 48)` in our usual row order. Keep the trained probabilities `(0.20, 0.20, 0.10, 0.50)` fixed. Its average held-out NLL is:

```math
\frac{18[-\log(0.20)]+22[-\log(0.20)]+12[-\log(0.10)]+48[-\log(0.50)]}{100}
\approx 1.2528.
```

The equal-probability model still scores 1.3863 because it assigns every record probability 0.25. It is a **baseline**: a simple alternative evaluated on the same observations. Our fitted model beats it on this new batch as well. That supports better prediction on this batch; it is not proof of superiority for every future factory or every type of observation. These counts are illustrative, not results of an experiment we ran.

If we also need to choose between training methods or settings, separate held-out data into two roles:

- **Training:** for example, use 60 records to fit each candidate model's parameters.
- **Validation:** compare the candidates on 20 different records and choose one. These records influence the choice, even though they did not fit the parameters directly.
- **Test:** evaluate the selected model once on 20 other records that did not influence either step.

The counts illustrate the roles, not a sample-size recommendation. If we inspect test results, change the model and inspect them again to decide whether to keep the change, those records are now helping us select the model. They are no longer an untouched final check.

### When fitting the training records is misleading

**Overfitting** means adapting to accidental details of the training data that do not predict fresh observations well. To see how this can happen even in a small table, consider a separate illustrative training set of only 10 records, with counts `(1, 1, 1, 7)`. Fitting its frequencies gives `(0.10, 0.10, 0.10, 0.70)`. Compare that fit with our earlier table, then evaluate both on 10 fresh records with counts `(2, 2, 1, 5)`:

| Candidate probabilities | Average NLL on the 10 training records | Average NLL on the 10 fresh records |
|---|---|---|
| Earlier table: `(0.20, 0.20, 0.10, 0.50)` | 1.0373 | **1.2206** |
| Fit to the small set: `(0.10, 0.10, 0.10, 0.70)` | **0.9404** | 1.3296 |

These scores use the same count-times-penalty calculation shown above. The small-set fit is better on its training data: it rewards the seven “neither” records with probability 0.70. But the fresh set has only five “neither” records and more blue records. The earlier table scores better there. Learning the small training sample more closely did not improve prediction on this fresh sample. With so few records, the particular scores are uncertain; the example demonstrates why we need a separate check.

**Generalization** means performing well on new observations from the population of interest. A training score measures how well we fit the examples already used; a held-out score supplies evidence about that new-observation performance. A nice-looking generated record supplies neither score.

**Exercise:** a model’s training NLL falls from 2.0 to 1.0 while NLL on the same held-out set rises from 2.1 to 2.4. Has learning improved?

**Worked answer:** lower NLL is better, so the model improved on its training records but worsened on the held-out records. We cannot claim improved generalization. This pattern is consistent with overfitting, but two scores alone do not establish the cause. For example, check that both held-out scores used the same records and scoring rule, and that those records represent the population we intend to generate from. If the held-out set was used to choose when to stop training, it serves as validation; a separate test set is still needed for the final check.

### Realistic individual outputs and coverage

**Fidelity** asks whether individual generated observations resemble real ones. **Coverage** asks whether the generator represents the different kinds of observations found in the population. In our record example, repeatedly returning “not blue and not large” produces a valid kind of record, but never produces the other three kinds. It fails coverage. Even a program storing the correct table can make this mistake if it always selects the highest-probability row instead of sampling.

Coverage alone is not enough: producing each of our four rows equally often covers all four, but makes “not blue and large” 25% of the outputs instead of the fitted 10%. Check both which outcomes appear and their frequencies.

To distinguish fidelity from coverage visually, imagine a map of eight islands, with real observations being locations on land. Nearby land locations form a **cluster**, or group of points. Generator A produces only locations on the first island. Its points look realistic individually, but seven islands are missing. Generator B produces some locations on every island, but also many in the sea. It reaches more islands, yet many individual outputs do not resemble the land observations. This is an analogy for checking generated data, not a claim about real geography.

**Exercise:** generator A puts all 100 generated locations on the first island. Generator B puts 10 on each of the eight islands and 20 in the sea. Which has better fidelity, which has broader coverage, and can either be called better overall from these facts alone?

**Worked answer:** A puts `100/100 = 100%` of its points on land, compared with B's `80/100 = 80%`: A has better fidelity by this simple rule. A reaches only `1/8` islands; B reaches `8/8`: B has broader coverage. There is no overall winner without stating how these goals matter for the application. Both were given 100 draws, so we did not make B's coverage look better simply by drawing more samples from it. Repeating the comparison with new draws would show how stable these counts are. We would also compare how often each island appears with its frequency in real observations.

**Exercise:** define a distribution, a loss, a sample and held-out evidence in one sentence each, using a numerical example from this lesson.

**Worked answer:**

- **Distribution:** our model assigns probabilities `(0.20, 0.20, 0.10, 0.50)` to the four colour–size combinations.
- **Loss:** its average negative log likelihood is about 1.2206 on the 100 records used to fit it; this scores the probabilities assigned to those observations.
- **Sample:** one random draw, such as 0.47, produces one record, “not blue and large”, without changing the probabilities.
- **Held-out evidence:** on the illustrative fresh counts `(18, 22, 12, 48)`, the unchanged model scores about 1.2528 versus the equal-probability baseline's 1.3863, supporting better prediction on that batch.

## Further reading

[Optional mathematics and sequence models](further-reading/01-mathematics-and-sequences.md) contains the detailed likelihood calculations, probability identities, expectations and Gaussian models, gradients and backpropagation, entropy and KL, and token generation with attention and teacher forcing. Their explanations and worked examples are preserved there. They are not part of the four introductory topics or required advance reading for this meeting.

Continue with [Lesson 02 — paper discussion and its reading assignment](02-generative-models.md).
