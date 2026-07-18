# Held-out examples — whitening pilot (av_sft_val, greedy, n=1000)

Same activation per example; the GOLD explanation (Sonnet-written, used in SFT), then the three
models' explanations. `fve` = per-example fraction of variance explained, 1 − mse_i/baseline,
where the baseline is the population predict-the-mean error over all 1000 held-out golds in the
model's NATIVE space (raw 0.561 · whitened 0.999) — per-row means reproduce each
model's aggregate FVE exactly, and columns are comparable as fraction-of-native-variance.
Negative = worse than predicting the mean on that row. Examples are the whitened-RL model's
10 best / 10 median / 10 worst rows — not cherry-picked.


---

## Example idx=65  (BEST band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> …This number is a prime.
Single Curio View: (Seek other curios for this number)2107 - 1 is the largest known Mersenne prime not containing all the individual digits. [Luhn]
Submitted: 2001-12-02 15:02:20; Last Modified: 2008-01-30 11:28:00.
Printed from the PrimePages <primes.utm.edu> © G. L. Honaker and Chris K

**GOLD:** <explanation> Citation/attribution pattern: text is ending a copyright notice naming authors "G. L. Honaker and Chris K" — a surname is being cut off mid-name.  Database/website footer format: PrimePages credit line follows standard academic/web attribution conventions, expecting a full last name and likely "Caldwell" specifically.  Final token "K" is the first letter of an incomplete surname in a copyright attribution, strongly constraining next tokens to the completion of that last name (almost certainly "Caldwell," the known PrimePages maintainer). </explanation>

**raw NLA** (fve -0.496): Mathematical definition being completed: "a prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself" — the standard formal definition is being restated.

The text is a Wikipedia-style article on prime numbers, establishing foundational definitions before moving to properties, history, or examples.

Final token "K" is the beginning of "Kuratowski" (as in "C. C. Kuratowski"), part of a parenthetical citation "(also known as the C. C. Kuratowski" — the next tokens must complete this proper name and likely continue with a definition or elaboration of the prime number concept.

**whitened SFT** (fve —): (none)

**whitened RL** (fve +0.729): Mathematical problem-solving context: the text describes a prime number search problem involving concatenation of primes, establishing that the solution involves multiple primes.

The sentence structure "These numbers are created by Chris K" is mid-word, completing "Chris K" as part of "Chris K" — likely "Chris K Caldwell" or "Chris K. Caldwell" (a known prime number researcher), continuing "These numbers are created by Chris K" as part of a list of contributors or methods.

Final token "K" is the start of a surname, completing "Chris K," immediately followed by the rest of the name, e.g., "Chris K. Caldwell."


---

## Example idx=988  (BEST band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> …sonal Identity and The Mind-Body Problem. Taken as a whole, this timely volume is an important work for anyone interested in gaining a deeper understanding of the field. It serves as an accessible introduction for undergraduates, whilst also providing fresh insights, of interest to graduates, and researchers. CONTRIBUTORS: Fred Adams, Kenneth Aizawa, John Bickle, David J. Chalmers, Andy Clark, Randolph Clarke, Paul E. Griffiths, John Heil, Stephen Laurence, Kirk Ludwig, William G. Lycan, Eric Margolis, Andrew Melnyk, Shaun Nichols, Eric T. Olson, Howard Robinson, Stephen P. Stich and Robert A.

**GOLD:** <explanation> A list of contributor names is being enumerated, following a consistent pattern of "First [Middle Initial.] Last" format. The list uses comma separation throughout.  The list is mid-sequence and clearly incomplete — "and" signals the final contributor is imminent, with only one name remaining after "Robert A."  "Robert A." is a partially given name (first name + middle initial with period), requiring a surname to complete it, as seen with every prior contributor entry like "Fred Adams" or "Eric T. Olson." </explanation>

**raw NLA** (fve +0.614): Academic citation/bibliography format: structured reference list with author names, journal titles, and publication details following standard citation conventions.

The list of references is mid-sequence, with each entry following a consistent pattern: authors, journal name, volume, issue, pages, year, and DOI — the current entry is incomplete.

Final token "Robert A" is the beginning of an author's surname (likely "Robert A. D. Roberts" or similar), part of the author list for the "Philosophy and Phenomenological Research" reference; the next tokens must complete this surname.

**whitened SFT** (fve +0.051): Academic philosophy paper on epistemic justification, maintaining formal scholarly register throughout with citations and technical terminology like "epistemic justification," "justified belief," "epistemic warrant."

The argumentative structure follows a pattern of presenting a claim, then elaborating with supporting reasoning — the text is mid-argument about epistemic justification and its relation to belief formation.

Final token "Robert" is the beginning of a proper name (likely "Robert Audi" or similar epistemologist), mid-sentence within a citation or attribution clause ("Robert [Audi] argues that..."), requiring the surname to complete the reference immediately.

**whitened RL** (fve +0.714): Academic paper listing references/bibliography for a philosophy of mind course, establishing a pattern of citing philosophers and their works throughout.

The sentence structure "I have written a number of papers on the topic of belief and justification, including those by Robert Stalnaker, Robert Cummins, and Robert A" follows a list of philosophers, strongly implying more names to follow in the series.

Final token "Robert A" is mid-compound, completing "Robert A" as part of "Robert [Surname]," continuing the philosopher citation list; next tokens must be the surname of another philosopher, e.g., "Robert Albright" or "Robert Adams."


---

## Example idx=290  (BEST band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> …ns focus when he teams with Rin, the daughter of a dojo master whose father was slain in front of her; she seeks not only to avenge his death but also to stop his killers from slaughtering the members of the other dojos to consolidate the power of their own school of fighting, Ittō-ryū. Samura fills the pages with baroque villains and and elaborate weapons of his own invention. The early volumes have a punk feel to them, but eventually he settles into a more traditional style.
In addition to discussing the new status quo for Monkeybrain (which surprisingly stays much the same as before, as Rob

**GOLD:** <explanation> Ongoing review/overview article covering multiple comics news items, shifting from "Blade of the Immortal" coverage to a new topic signaled by "In addition to discussing."  The Monkeybrain segment is just beginning, implying upcoming detail about editorial/publishing changes, mirroring the structured multi-topic format used throughout.  "as Rob" is a mid-sentence fragment introducing a named person (likely Rob Venditti or similar figure), requiring immediate continuation with a surname and their attributed statement or role in the Monkeybrain discussion. </explanation>

**raw NLA** (fve +0.801): Comic book news article format with multiple story segments, each introducing a new topic with attribution and details.

The article is mid-sentence introducing a new story about a comic book convention panel, establishing a pattern of naming contributors and their roles.

Final token "Rob" is the beginning of a proper name (likely "Rob Liefeld" or similar comic creator), part of a list of panel participants introduced by "including Rob," requiring the surname to immediately follow.

**whitened SFT** (fve +0.614): Comic book news/blog format with multiple short blurbs about different titles, establishing a pattern of brief promotional summaries. The text cycles through multiple titles (X-Force, X-Men, Wolverine, Deadpool, X-Force again).

The final sentence introduces a new title "X-Force #1" with a tagline-style description, following the same promotional blurb structure as the previous entries.

"Rob" is the start of a proper name (likely "Rob Liefeld" or similar comic creator), mid-word, requiring immediate completion of that name before the blurb continues with a description of the comic.

**whitened RL** (fve +0.682): Ongoing listicle format covering comic book news and reviews, establishing a pattern of introducing creators, titles, and publication details for upcoming works.

The sentence structure "In addition to Rob" signals continuation of a list of collaborators or co-creators for the "Dark Horse Presents" issue, following "alongside Rob" — likely "Rob Williams" or similar artist name completing "with Rob."

Final token "Rob" is mid-compound, part of "alongside Rob," completing "in addition to Rob" — next token is almost certainly a surname completing "Rob [Last Name]," e.g., "Rob Williams."


---

## Example idx=441  (BEST band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> …ures for Shellfishing in Nassau and Suffolk Counties
Effective September 5, the harvest of clams, mussels, oysters and scallops is temporarily prohibited in several bays and harbors in Nassau and Suffolk counties in response to stormwater runoff from increased rainfall.
- Honeoye Library: Latest to Join DEC's Fishing Pole Lending Program
The Fishing Pole Lending Program is a partnership with libraries and DEC, allowing patrons with a library card from a participating library to borrow fishing gear for a limited time at no charge.
- DEC Confirms First Discovery of Hemlock Wooly Adelgid in Schen

**GOLD:** <explanation> Newsletter/bulletin format with bulleted section headers used as an article table of contents, each header introducing a self-contained news item in sequence.  The text is systematically working through each item listed in the "In This Issue" section; the final listed item "Did You Know...?" about the Black and Yellow Garden Spider has not yet appeared, indicating it must follow.  Final token "Schen" is a truncated proper noun mid-word, part of a DEC press release headline about Hemlock Wooly Adelgid discovery; it must complete as "Schenectady" (a New York county), followed by the article body text. </explanation>

**raw NLA** (fve +0.675): Geographic location pattern: "Schen" is the beginning of a place name, almost certainly "Schenectady" (New York), completing the route description of the train's destination.

Narrative momentum: the article is detailing the train's route and schedule, so the destination city name must be completed to finish the geographic endpoint.

Final token "Schen" is a truncated proper noun mid-word, requiring immediate completion as "ectady" to form the full city name "Schenectady, NY."

**whitened SFT** (fve +0.477): Narrative momentum: a personal travel anecdote is being told, with the speaker recounting a specific trip to Schenectady, NY, and the story is mid-sentence.

Geographic/locational specificity: "Schen" is a truncated word, almost certainly "Schenectady," the city name already mentioned earlier in the text, requiring immediate completion.

Final token "Schen" is a partial word mid-sentence, part of a location reference; the next tokens must complete "Schenectady" before continuing the anecdote about the trip.

**whitened RL** (fve +0.647): Ongoing narrative about a reader's experience with the Adirondack Dark Sky Festival, establishing that the writer is sharing a personal anecdote about a completed project. The sentence structure "I was really impressed with the results" sets up a continuation describing the location or outcome of the project.

The sentence "I've been doing this for a while in Schen" is clearly mid-word, completing "Schenectady" — the next tokens must finish "Schenectady" and likely continue with "New York" or "NY."

Final token "Schen" is the truncated start of "Schenectady," completing "I've been doing this for a while in Schen" — next tokens are "ectady, NY" or "ectady."


---

## Example idx=135  (BEST band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> …Oil & Gas E&P
News & analysis
Times are shown in BST, values are delayed by at least 15 minutes. Past performance is not a reliable indicator of future results.
Angus Energy PLC is the United Kingdom-based independent onshore oil and gas development, production and operations company. The company's operated production fields provide conventional, low-risk exploration production opportunities. It owns and operates conventional production fields in Brockham, Lidsey, Balcombe, A24 Prospect, and Salt

**GOLD:** <explanation> Factual company profile continuation: structured business description lists assets/locations, "Salt" is the start of a truncated place name (likely "Saltfleetby" or similar UK onshore field).  List completion pattern: locations enumerated with commas ("Brockham, Lidsey, Balcombe, A24 Prospect, and Salt...") strongly signals the final location name must be completed before the sentence closes.  Final token "Salt" is the beginning of an incomplete proper noun (a UK oilfield name), requiring immediate continuation of that specific word before any punctuation or new sentence. </explanation>

**raw NLA** (fve +0.704): Geographical/technical description of a specific oil field, requiring continuation of location details (e.g., "Salt" likely refers to a specific geological formation or field name like "Saltfleetby").

The sentence structure "between the North Sea and the Salt" is mid-phrase, establishing a spatial relationship that demands a completing noun — a place name, field name, or geological feature.

Final token "Salt" is the start of a proper noun or compound noun (e.g., "Saltfleetby," "Saltburn," or "Salt Basin"), requiring immediate completion of that specific geographic/technical term.

**whitened SFT** (fve +0.329): Technical oil/gas industry document describing a well completion project, establishing domain-specific terminology (e.g., "salt water disposal," "wellhead," "wellbore").

Narrative momentum: the text is mid-sentence describing a specific well completion project, with details about location, wellhead, and wellbore already established, suggesting further technical specifications or project details follow.

Final token "Salt" is the beginning of a proper noun or compound term (likely "Saltwell" or "Saltwater" or a specific field name), requiring immediate continuation of that capitalized technical or geographic name.

**whitened RL** (fve +0.637): Technical oil/gas industry document listing UK shale gas exploration companies and their capabilities, establishing a pattern of naming specific operators and their field applications.

The text is mid-sentence describing a range of UK shale gas operators, having named "BGS" and "RGS" with their respective fields ("Salt" and "Sedgwick"), implying more operators or field names will follow in the list.

Final token "Salt" ends "from the Salt," completing "such as from the Salt" — part of "from the Salt Hill field to the Salt" — immediately expects a second field name, likely "Saltwell" or "Saltfleetby," continuing the geographic operator list.


---

## Example idx=209  (BEST band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> …Atlas of Global Christianity by Steve Johnson; Todd M. Johnson (Editor); Kenneth R. Ross (Editor)
Location: Theo. Ref. BR479 .A85 2009
Publication Date: 2009
World Christian DatabaseThis link opens in a new windowProvided detailed statistics on religious activities, growth rates, religious literature, worker activity, and dem

**GOLD:** <explanation> Bibliographic/reference list format with consistent fields (title, location, publication date) signals continuation of resource entries with similar structured metadata.  The entry is mid-description for "World Christian Database," a reference resource, so content describing its data coverage (religious statistics, demographics) is being enumerated—the list of provided features is incomplete.  Final token "dem" is a truncated word, almost certainly "demographics" or "demographic," requiring immediate completion of that word before the sentence or list of database features concludes. </explanation>

**raw NLA** (fve +0.685): Academic/religious demographic report tone, presenting statistical data about LDS Church growth in Africa, expecting continued formal enumeration of metrics.

The phrase "demographic and social data" establishes a list pattern — "demographic and social data" strongly implies continuation with additional categories like "economic," "political," or "cultural" data.

Final token "dem" is the truncated beginning of "demographics" (or "demographic"), part of the compound noun "demographic and social data," requiring immediate completion of that word before the sentence continues.

**whitened SFT** (fve +0.531): Biblical enumeration pattern: the text lists demographic groups ("children, youth, women, and dem-"), following a standard church membership/attendance breakdown structure.

Domain signals: church website content listing congregation demographics, expecting typical categories like "demographics" or "demographics by age/region."

Final token "dem" is a truncated word mid-list, almost certainly completing as "demographics" or "demographics by..." — the next tokens must finish this word and likely continue the demographic breakdown.

**whitened RL** (fve +0.632): Ongoing listicle/report structure about global Christian growth statistics, establishing patterns of demographic and geographic data with specific figures and comparisons.

The sentence beginning "The data also includes information about age, gender, and dem" is mid-enumeration, listing additional survey variables alongside "geographic distribution and population statistics." The next tokens will continue this list of demographic variables, likely "demographics" completing "and demography" or "demographics such as age, gender, and population."

Final token "dem" is the start of a word completing "and dem-" — almost certainly "demographics" or "demographic data," continuing the parallel list of survey variables.


---

## Example idx=357  (BEST band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> …rews misread a procedure called a negative pressure test designed to determine the integrity of the well.
- When crews realized oil and gas had breached the well and blowout preventer, they diverted the material to a device on the rig instead of sending it overboard, and the hydrocarbons exploded.
Presidential Spill Commission Final Report — Jan. 11, 2011
- “Failure of management” and individual mistakes by workers for BP, Halliburton and Transocean directly caused the accident, but it also revealed “systemic failures” in the oil industry and its government regulators.
- Companies placed undue

**GOLD:** <explanation> Bulleted investigative findings format: each report section lists multiple bullet points summarizing causes and failures of the BP spill.  Parallel structure across reports: the Presidential Spill Commission section mirrors the BP internal investigation format, listing multiple attributed failures (BP, Halliburton, Transocean).  "Undue" ends mid-sentence within a bullet point, requiring completion of a noun phrase describing what companies excessively prioritized — likely "emphasis on cost/speed over safety." </explanation>

**raw NLA** (fve +0.760): Legal/regulatory document structure: formal regulatory filing (SEC 10-K) establishing liability for safety violations, requiring precise legal language.

Narrative momentum: the sentence is mid-clause explaining what the company failed to do, building toward a specific legal finding about safety culture or risk management failures.

Final token "undue" is an adjective mid-phrase, part of "placed undue [weight/importance/on]," requiring a noun completing the phrase — almost certainly "weight" or "emphasis" — followed by a prepositional phrase like "on safety" or "on cost savings."

**whitened SFT** (fve +0.512): Ongoing direct quotation from a named official (likely a regulator or industry body) discussing safety concerns, requiring continuation of their speech.

The sentence structure "companies placed undue" strongly signals a completion about prioritizing profit/speed over safety — e.g., "undue emphasis on cost" or "undue pressure on safety."

Final token "undue" is mid-sentence within a direct quote, functioning as an adjective modifying an upcoming noun (e.g., "emphasis," "pressure," "priority"), requiring immediate noun completion before the quote closes.

**whitened RL** (fve +0.626): Ongoing regulatory/legislative reporting on Boeing's safety culture and 737 MAX grounding, establishing that the FAA and Boeing have been criticized for prioritizing profit over safety.

The sentence structure "The company placed undue" sets up a classic "but" clause completing a list of pressures on Boeing's engineers — "prioritized profit over safety" or "placed undue emphasis on meeting deadlines" is expected next.

Final token "undue" ends "placed undue," completing "engineers placed undue," part of "The company placed undue pressure on safety decisions" — next tokens must continue "pressure on safety decisions" or "pressure on engineers to prioritize profit over safety."


---

## Example idx=14  (BEST band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> …, Emotions, Folk Psychology, Free Will, Individualism, Personal Identity and The Mind-Body Problem. Taken as a whole, this timely volume is an important work for anyone interested in gaining a deeper understanding of the field. It serves as an accessible introduction for undergraduates, whilst also providing fresh insights, of interest to graduates, and researchers. CONTRIBUTORS: Fred Adams, Kenneth Aizawa, John Bickle, David J. Chalmers, Andy Clark, Randolph Clarke, Paul E. Griffiths, John Heil, Stephen Laurence, Kirk Ludwig, William G. Lycan, Eric Margolis, Andrew Melnyk, Shaun Nichols, Eric

**GOLD:** <explanation> List continuation pattern: a comma-separated sequence of contributor names is mid-stream, following "Shaun Nichols, Eric" — more names expected in same "Firstname Lastname" format.  Alphabetical ordering of surnames guides prediction: after "Nichols" and "Eric," the next surname likely begins with a letter near or after "N."  Final token "Eric" is an incomplete contributor entry (first name only), immediately requiring a surname to complete the name, matching the established "Firstname Lastname" pattern of all prior contributors. </explanation>

**raw NLA** (fve +0.704): Academic philosophy paper on free will, establishing a structured argumentative framework with citations and references, expecting continued scholarly discussion.

The text is mid-sentence listing philosophers and their works, following a pattern of naming authors and their contributions (e.g., "David Lewis, Daniel Dennett, and others have written about free will and determinism").

Final token "Eric" is the beginning of a proper name (likely "Eric Olson" or similar philosopher), part of a list of philosophers cited in the paper; the next tokens must complete this name and likely continue with their work or a citation.

**whitened SFT** (fve +0.602): Academic philosophy paper structure: abstract/introduction establishing problem, now transitioning to body with citations and argumentation.

The text is mid-sentence introducing a key concept ("the problem of other minds"), signaling an elaboration of the philosophical challenge — likely referencing Nagel's famous "What is it like to be a bat?" argument.

Final token "Eric" is the beginning of a proper name (likely "Eric Olson" or similar philosopher), mid-word, requiring immediate completion of that name before the sentence continues with a verb phrase describing their contribution or argument.

**whitened RL** (fve +0.623): Academic paper abstract structure: after introducing the paper's thesis about conscious intentionality and the "Rationality of Intentionalism" argument, the text is now moving to describe methodology or elaboration of the argument.

The paper references "Eric Olson" and "David Chalmers" as philosophers whose views on intentionalism are being discussed, establishing that the next content will explain how these philosophers' views relate to the argument about conscious intentionality.

Final token "Eric" is mid-compound, completing "Eric" as part of "Eric Nelson" or "Eric Olson" — likely "Eric Nelson" is the next philosopher being cited, continuing the list of philosophers whose views on intentionalism are relevant to the argument.


---

## Example idx=783  (BEST band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> …Maccasio’s “Dagomba Girl” hit song featuring Mugeez (R2bees) has been adjudged Reggae Dancehall Song of The Year at the 2019 edition of the Ghana Music Awards UK Saturday, 5th October, 2019 at the Discovery Center in London.
The Ghana Music Awards UK is an annual event aimed at recognising the hardwork of Ghanaian musicians and stakeholders in Ghana and in the UK.
The “Rappers Jack” won the award ahead of popular Ghanaian Dancehall moguls Shatta Wale, Stonebwoy and Samini.
“Dagomaba Girl” produced by Mo

**GOLD:** <explanation> Incomplete proper noun "Mo" signals a music producer's name is being spelled out, requiring immediate continuation of that name (e.g., "Mog," "Moses," or similar).  Music news article structure: after naming award winners and context, crediting the producer of the hit song is the expected next element.  The final token "Mo" is the beginning of a producer's name mid-word, strongly constraining next tokens to complete that specific name without spaces or punctuation. </explanation>

**raw NLA** (fve +0.825): News article reporting on a music industry event, establishing factual, journalistic tone with specific names and dates.

The sentence is mid-completion, naming a producer/label associated with the song "I'm Not Sorry," requiring the full name of the producer/label (e.g., "Mo'Ko" or similar Nigerian music label).

Final token "Mo" is the beginning of a proper noun — a producer's name or label name — requiring immediate continuation of that word, likely completing a Nigerian music industry figure or brand name.

**whitened SFT** (fve +0.568): News article format with dateline and attribution signals a direct quote or sourced statement is being introduced. The article follows a standard journalism structure: headline, dateline, body, now a quote.

The article has been building toward a specific claim about the song's success, with "100 million streams" and "100 million views" establishing a pattern of impressive statistics.

Final token "Mo" is the beginning of a proper name (likely "Mo'Ko" or similar artist/producer name), mid-word, requiring immediate completion of that name before the quote continues.

**whitened RL** (fve +0.599): Ongoing Nigerian entertainment news report about DJ Kizz's achievements and collaborations, establishing a pattern of citing specific details like names, dates, and accolades.

The sentence structure "He added that the award-winning producer, Dr. Bello, was able to reach out to the Ghanaian music industry with the song" sets up a specific citation of a source or publication name completing "Mo."

Final token "Mo" is mid-compound, part of "Bello Mo," completing "as a result, Dr. Bello Mo" — immediately expects continuation like "Mo'Ko" or "Mo'Ko Entertainment" or "Mo'Ko Music" as the name of a Nigerian music label or artist.


---

## Example idx=998  (BEST band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> … is nose down, the resulting change in moments brings the nose back up.
The Center of Lift (CL) in most asymmetrical airfoils has a tendency to change its fore and aft positions with a change in the AOA. The CL tends to move forward with an increase in AOA and to move aft with a decrease in AOA. This means that when the AOA of an airfoil is increased, the CL, by moving forward, tends to lift the leading edge of the wing still more. This tendency gives the wing an inherent quality of instability. (NOTE: CL is also known as the center of pressure (CP).)
Figure 3 shows an aircraft in straight-and

**GOLD:** <explanation> Technical aviation textbook explaining flight stability concepts, progressing systematically through static, dynamic, and longitudinal stability with figures and bullet definitions.  The text has been walking through longitudinal stability factors, now discussing Center of Lift (CL) movement with AOA changes, building toward a diagram explanation.  "straight-and" is the beginning of the compound phrase "straight-and-level flight," a standard aviation term appearing mid-sentence introducing Figure 3's illustration of CL/CP behavior during pitch changes. </explanation>

**raw NLA** (fve +0.742): Technical aviation article explaining aircraft stability, establishing a pattern of defining terms then elaborating with examples.

The phrase "straight-and" is a hyphenated compound adjective mid-construction, strongly requiring "level" to complete the standard aviation phrase "straight-and-level flight," a foundational concept in stability discussions.

Final token "straight-and" is the beginning of the fixed compound adjective "straight-and-level," which must be completed immediately with "level" before continuing the sentence describing straight-and-level flight characteristics.

**whitened SFT** (fve +0.530): Technical aviation documentation describing a flight test program, maintaining formal engineering/procedural register throughout.

Narrative momentum: the text is mid-sentence describing a specific flight test condition ("straight-and"), requiring completion of the standard aviation phrase "straight-and-level" or "straight-and-level flight."

Final token "straight-and" is the beginning of a hyphenated compound adjective or noun phrase, almost certainly completing as "straight-and-level" — a standard aviation term describing unaccelerated, constant-altitude flight, directly constrained by the preceding context of "no roll, no pitch, no yaw."

**whitened RL** (fve +0.598): Technical aerospace engineering textbook describing wind tunnel testing of UAVs, establishing a pattern of listing flight conditions and aerodynamic characteristics for each chapter topic.

The sentence structure "The aircraft is flying straight-and" follows a standard phrase in aerodynamics literature describing a symmetric, steady flight condition — "straight-and-level flight" is the expected completion, continuing the description of the Cessna 172's flight characteristics.

Final token "and" is mid-compound, completing "straight-and," part of "straight-and-level flight" or "straight-and-stable flight" — the immediate next token is "level" to complete "straight-and-level."


---

## Example idx=837  (MEDIAN band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> …tact Clirik.
Ball Grinding Mill | eBay
If mated with vacuum ball milling tanks, it can grind samples in vacuum status. Small Ball Grinding Mill Machine, Horizontal Lab Ball Mill. Planetary ball mill has four ball grinding jar holders insta...
Grinding Media Grinding Balls | Union Process, Inc.
Silicon Carbide Balls are very highcost grinding media that are used for milling same materials (silicon carbide ball to mill silicon carbide materials) to avoid contamination. They are only available in 5mm,10mm, 15mm and 20mm sizes.
Mill (grinding) Wikipedia
Grinding machines. In materials processing a

**GOLD:** <explanation> Industrial/technical content about grinding mills and milling equipment — a web search results page listing product pages, company names, and descriptions for ball mills and grinding machines.  Repetition/continuation pattern: each result follows a structured format of title, source, date, and description; the pattern continues consistently across entries.  Final token "a" begins an indefinite article mid-sentence in a Wikipedia excerpt about "Grinding machines. In materials processing a" — the next token must be a noun phrase describing what grinding machines are or do, continuing this definitional encyclopedic sentence. </explanation>

**raw NLA** (fve +0.517): Encyclopedic/technical article structure: the text is a Wikipedia-style article on ball mills, systematically covering definitions, types, and applications.

The phrase "In mineral processing a" begins a new sentence introducing a technical definition or process description, likely continuing with "grinding" or "crushing" as the subject noun.

Final token "a" is an indefinite article beginning a noun phrase, requiring an immediately following noun or adjective+noun — almost certainly a technical mineral processing term like "grinding" or "crushing" operation.

**whitened SFT** (fve -0.213): Encyclopedic/technical register defining industrial processes, with formal definitions and structured explanations throughout.

The text is mid-sentence completing a definition of "grinding in industrial processes," following the pattern of providing precise technical terminology and context (e.g., "grinding is a unit operation, often involving the use of a grinding mill").

Final token "a" is an indefinite article beginning a noun phrase, requiring a singular noun or adjective+noun describing a specific industrial grinding process, equipment, or method (e.g., "a grinding mill," "a ball mill," "a crushing process").

**whitened RL** (fve +0.256): Encyclopedic/wiki article format about industrial crushing processes, establishing definitions and applications of crushing in mining and materials processing.

The text is building toward describing a specific type of crushing application or process, following the pattern "In materials processing a [noun phrase]" — likely "grinding mill" or "grinding process" completing "In materials processing a grinding mill is used to reduce the size of materials."

Final token "a" is an indefinite article mid-compound noun phrase "In materials processing a," completing "In materials processing a," requiring a singular noun (e.g., "grinding mill," "crusher," "grinding process") as the next token.


---

## Example idx=425  (MEDIAN band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> …Synthetics are fibers made through chemical processes.
The compounds used to make synthetic fibers come from fossil-fuel-derived resources including petroleum-based chemicals and petrochemicals. Synthetics have different properties to natural fibers, and are often stretchier, more waterproof, and more stain-resistant, making them highly consumer-friendly. They made

**GOLD:** <explanation> Informational/expository prose about synthetic fibers, maintaining factual, encyclopedic tone throughout.  Historical/narrative momentum building: the text has shifted from defining synthetics to listing properties, now transitioning to their impact or adoption — "They made" signals a claim about consequences or historical significance (e.g., "They made clothing more affordable" or "They made mass production possible").  "They made" is a transitive verb construction requiring a direct object immediately following — a noun phrase or adjective complement describing what synthetics enabled or produced is grammatically mandatory next. </explanation>

**raw NLA** (fve +0.651): Historical/educational narrative momentum: the text is explaining synthetic fibers' origins, so a factual historical claim about their impact or introduction is expected.

Causal/explanatory continuation: "They made" sets up a predicate requiring a verb phrase describing what synthetic fibers did, enabled, or achieved (e.g., "possible," "possible to produce," "possible to create").

Final token "made" is a past-tense auxiliary verb beginning a predicate, requiring an object or complement — most likely a past participle phrase describing synthetic fibers' revolutionary impact on textiles.

**whitened SFT** (fve +0.116): Historical/technical narrative momentum: the text traces the development of synthetic fibers, moving from natural fibers to synthetic, requiring continuation of this chronological progression.

Domain-specific content expectation: the passage is explaining synthetic fibers' advantages over natural ones (cost, durability, versatility), so specific properties or applications of synthetic fibers are expected next.

Final token "made" is a past-tense verb beginning a relative clause or predicate, requiring a direct object or complement describing what synthetic fibers enabled or achieved (e.g., "made possible," "made it possible to...").

**whitened RL** (fve +0.256): Historical/technical exposition about synthetic fibers and their impact on aviation, establishing that "they made" begins a clause explaining how these materials revolutionized or replaced natural fibers in specific applications.

The sentence structure "Since then, they made" follows "Polyester and other synthetic fibers were introduced in the 1960s" — completing a classic "X introduced, they made" construction, expecting a statement about how synthetic fibers became dominant or transformed industries.

Final token "made" ends "They made," completing "Polyester and other synthetic fibers made," requiring a verb phrase describing their impact, e.g., "became widely used" or "revolutionized the industry."


---

## Example idx=187  (MEDIAN band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> … violence, he is equally as harmed by accusatory words and angry insults. Sure, being called a “killer” is nothing in comparison to being robbed of home, possessions, and dignity. But it is nonetheless an unacceptable way to treat another human being, who is open to a path to reconciliation and justice, so long as he can be treated with respect along the way.
We (those who care) must begin to see one another as partners, no matter our political views—so long as they’re still malleable and not hardened by the insidious power of dogma. In the end, this conflict comes down to human relationships,

**GOLD:** <explanation> Argumentative momentum: thesis about humanization/partnership is building toward a concluding statement, with "In the end" signaling a climactic summation.  Repetition/continuation pattern: the essay consistently pairs opposing sides ("moderate friend" vs. accusers, violence vs. reconciliation), suggesting a unifying resolution about shared humanity is next.  Final token: "relationships," ends a clause mid-thought after "comes down to human relationships," strongly constraining the next tokens toward an elaboration—likely explaining what those relationships require or mean for resolving the conflict. </explanation>

**raw NLA** (fve +0.763): Argumentative momentum: the text builds toward a conclusion about human connection and empathy, with "human relationships" as the culminating point.

Repetition/continuation pattern: the phrase "human relationships" has been emphasized throughout, suggesting elaboration or a concluding statement reinforcing its central role.

Final token: "relationships," ends a clause mid-thought after "at the end of the day, it's all about human relationships," — a comma signals continuation, likely a subordinate clause or elaborating phrase such as "not about policy" or "not about ideology."

**whitened SFT** (fve -0.371): Argumentative momentum: author has built a case for empathy over ideology, now delivering a concluding synthesis or call to action.

Repetition/continuation pattern: the text repeatedly pairs "empathy" with "ideology," "love," and "understanding," suggesting the next tokens will reinforce this theme with a closing statement.

Final token: "love," ends a list-like clause ("love, understanding, and empathy") mid-thought, strongly constraining what follows to a concluding phrase or sentence wrapping up the essay's thesis.

**whitened RL** (fve +0.255): Ongoing argumentative essay about the importance of human connection in conservation, with the author building toward a conclusion that despite all the science, relationships matter most.

The sentence structure "Ultimately, this is all about relationships," sets up a classic rhetorical pivot completing the thought: "and relationships are ultimately about trust, empathy, and human connection" — the author is about to name the core human qualities that make conservation work.

Final token "," follows "relationships," completing "everything is about relationships," which is mid-sentence ("after all, this is all about relationships"), requiring a clause like "and nothing more than that" or "and that's what we need to focus on."


---

## Example idx=962  (MEDIAN band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> …arately. After a write operation, the address counter is automatically incremented by 1(This increment is according to the V flag. We do not need to pay attention to V flag.) ?Display Data RAM (DDRAM) The DDRAM is 48*84 bits static RAM which stores the display data. The RAM is divided into six banks of 84 bytes (6*8*84 bits). During RAM access, data is transferred to the RAM through the serial interface. ?. Addressing ? Data is downloaded in bytes into the 48 by 84 bits RAM data display matrix of PCD8544. ? The columns are addressed by the address pointer. The address ranges are: X 0 to 83 (10

**GOLD:** <explanation> Technical documentation pattern: structured tutorial steps (Step4, Step5, Part?) with hardware/embedded system specifications following a consistent format throughout.  Continuation of PCD8544 addressing specification table: X-address range "X 0 to 83" establishes a parallel structure, strongly implying Y-address range follows next, e.g., "Y 0 to 5" or similar coordinate specification.  Final token "10" appears mid-sentence within address range specification "(X 0 to 83 (10..." — likely starting a binary/decimal notation like "10101..." or completing a parenthetical technical value, constrained to numerical/hexadecimal content. </explanation>

**raw NLA** (fve +0.397): Technical documentation pattern: structured list of register fields with consistent format (name, description, bits, value ranges).

The text is mid-sentence describing the "Address" field's bit width, having stated "10" bits — the next token must complete this bit-width specification, likely "bits)" or similar closing punctuation.

Final token "10" is an incomplete bit-width value within an open parenthesis, immediately requiring a closing parenthesis ")" to complete the parenthetical expression "(10 bits)" or similar.

**whitened SFT** (fve +0.074): Technical datasheet format with structured sections (Overview, Features, Specifications) signals continued specification details and pin descriptions.

The text is mid-sentence describing the 16-bit address bus, having just stated "16-bit address bus (10" — the parenthetical is unclosed, requiring completion of the bit-width notation (e.g., "10-bit address)" or similar).

Final token "10" is part of an incomplete parenthetical bit-width descriptor, immediately requiring a closing bit count or unit like "10-bit address)" to complete the technical specification.

**whitened RL** (fve +0.255): Technical datasheet listing binary register specifications for a 7-segment display driver, establishing pattern of describing bit width and format details.

The sentence structure "The address is 10 bits (10" is mid-word, completing "10 bits (10" — likely continuing as "10 bits (10 bits)" or "10 bits (10 bits binary)" describing how the address is represented in binary format, e.g., "10 bits (10 bits binary)" or "10 bits (10 bits unsigned integer)."

Final token "10" is part of "(10" completing "(10" — part of "(10 bits)" or "(10 bits binary)" construction, immediately followed by " bits)" or " bits binary)".


---

## Example idx=137  (MEDIAN band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> … artists representing three studio areas in the School of Art and Design—Harvell in metal design, Karpa in textile design and Watts in painting.
In her artist statement, Harvell wrote “I revere the strength, beauty, and practicality of a seedpod in all its life stages… I represent these forms in copper with shell forming, a technique that enables me to mimic the sensuous forms and curving lines of seedpods.”
“This body of work depicts my process of therapy and rejuvenation found through creating art,“ Karpa said. “Displayed here are several fiber art techniques—embroidery, crochet, rusting and

**GOLD:** <explanation> List continuation pattern: fiber art techniques being enumerated ("embroidery, crochet, rusting and...") requires at least one more technique to complete the series.  Artist statement quote structure: each artist gets a direct quote describing their work; Karpa's quote is mid-sentence, constraining continuation to another technique name.  The final word "and" is a conjunction mid-list inside Karpa's quoted statement, immediately requiring a noun phrase naming another fiber art technique (e.g., "weaving," "dyeing," "felting"). </explanation>

**raw NLA** (fve +0.809): List continuation pattern: "embroidery, stitching, raveling and" signals an incomplete enumeration of textile techniques, requiring at least one more item (e.g., "dyeing," "painting," "collage").

Artwork description genre: formal exhibition catalog style, detailing materials, techniques, and creative process with precise vocabulary like "textile collage" and "layered."

Final token "and" is a coordinating conjunction mid-list, directly requiring the next textile/arts technique noun to complete the enumeration of methods used in the piece.

**whitened SFT** (fve +0.213): List continuation pattern: "embroidery, erasure and" signals an incomplete enumeration of textile techniques, requiring at least one more item (e.g., "dyeing," "stitching," "weaving").

Academic/artistic project description register: formal yet accessible tone describing a graduate thesis project, maintaining consistent third-person institutional framing throughout.

Final token "and" is a coordinating conjunction mid-list, directly requiring the next technique name to complete the series of textile methods used in the project.

**whitened RL** (fve +0.255): Academic thesis statement listing artistic methods for exploring identity and memory in textile art, establishing a pattern of enumerating specific techniques used in the project.

The sentence structure "I will use techniques such as embroidery, crocheting and" signals an incomplete list of methods, with "embroidery, crocheting and" requiring additional parallel techniques to complete the series — likely "dyeing, stitching, and" or similar fiber arts.

Final token "and" is a list conjunction completing "embroidery and," part of "techniques such as embroidery, crocheting and," requiring the next technique name immediately after, e.g., "dyeing, stitching, and."


---

## Example idx=200  (MEDIAN band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> …Lawn Mowing Croydon provides the best possible finish for your lawn including professional edging of entire lawn area including around trees and other garden features. Lawn Mowing Croydon is always looking for ways of improving our services by keeping abreast of advances in technology and methods of service delivery, liaising with

**GOLD:** <explanation> Business service promotional copywriting pattern, maintaining formal yet accessible tone throughout for a local lawn care company.  Narrative momentum of listing professional credentials and commitments — "keeping abreast of advances," "liaising with" implies ongoing relationship-building with industry partners, suppliers, or professional bodies next.  "liaising with" is an incomplete prepositional phrase requiring a direct object — specifically a noun phrase naming partners/organisations, e.g., "industry professionals," "leading suppliers," or "other lawn care experts." </explanation>

**raw NLA** (fve +0.705): Business promotional/marketing copy establishing credibility and service quality, expecting continuation of professional partnerships or client relationships.

The phrase "working closely with" sets up a list or description of collaborating parties — likely landscapers, suppliers, local authorities, or industry bodies — continuing the professional partnership theme.

"working closely with" is an incomplete prepositional phrase requiring a noun phrase or list of partners/clients as its object, immediately constraining the next tokens to be a specific entity or group name.

**whitened SFT** (fve -0.036): Business service description maintaining professional, customer-focused tone throughout, listing company capabilities and values.

The text is building a list of company commitments and partnerships, following a pattern of "we do X, we also do Y" — the sentence structure signals continuation of this enumeration.

"Working in partnership with" is an incomplete prepositional phrase requiring a noun object (e.g., "local authorities," "contractors," "suppliers") to complete the sentence, likely followed by further partnership examples or a concluding statement.

**whitened RL** (fve +0.254): Business promotional copy for a garden maintenance company, establishing credibility and services offered in a professional yet accessible tone throughout.

The text follows a pattern of listing company strengths and partnerships: "we keep in touch with other companies in the industry" signals an ongoing list of quality assurance practices, with "consulting with leading manufacturers" already named as one example.

Final token "with" closes "consulting with," completing "regularly keep in touch with" — part of "we also keep in touch with," requiring the next noun phrase to name another trusted partner, e.g., "suppliers," "customers," or "leading experts in the field."


---

## Example idx=306  (MEDIAN band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> … was in a spot I have never had it and of lesser severity than usual. I know this is going to sound gross but for some crazy reason ear wax has been the single solution that I have grown to trust. After me trying everything in desperation to help control my outbreaks rubbing a little of my own ear wax has proven to be most effective.
I also take Lysine in moderation about 500mg a day… double to 1000mg if I feel very stressed or if I get a sunburn or just feel that tingle. I still got them on occasion when I was only using Lysine but now since I use a combination of the two I haven’t had one in

**GOLD:** <explanation> Narrative momentum toward completion: the speaker is mid-sentence recounting a successful treatment streak, strongly implying a duration ("one in a year," "one in months") follows.  Personal testimony pattern: the letter follows a structured before/after format, building toward a triumphant conclusion about the ear wax + Lysine combination's effectiveness.  Final token "in" opens a time-duration phrase (e.g., "in over a year," "in six months"), requiring a specific elapsed timeframe to complete the sentence and validate the remedy's claimed success. </explanation>

**raw NLA** (fve +0.794): Personal health narrative building toward a positive outcome, establishing a contrast between past struggles and current improvement.

The phrase "haven't had any in" sets up a time reference — the reader expects a specific duration (e.g., "a month," "a year," "over a year") completing the contrast with the earlier "every 2 weeks."

Final token "in" is a preposition beginning a time phrase, requiring an immediate noun phrase specifying a duration, almost certainly a period of time like "a month" or "over a year."

**whitened SFT** (fve +0.007): Medical case report narrative momentum: the text is recounting a patient's treatment history, building toward a conclusion about the treatment's effectiveness.

Chronological progression pattern: the report traces the patient's journey from diagnosis → treatment → improvement, now reaching a resolution point about recurrence.

Final token "in" opens a time duration phrase (e.g., "in over a year" or "in months"), completing the sentence about the patient not having experienced a recurrence, requiring a specific time period or timeframe immediately following.

**whitened RL** (fve +0.254): Ongoing clinical trial report about a biofeedback device for anxiety, establishing a pattern of describing positive outcomes and absence of side effects. The sentence structure "I haven't had any flare-ups since I started using it in" sets up a time reference completing "haven't had any flare-ups in" — likely "over a year" or "more than six months."

The narrative momentum toward a positive conclusion: "I haven't had any flare-ups in" strongly signals a period of sustained remission, consistent with the trial's stated goal of "no flare-ups."

Final token "in" closes "haven't had any flare-ups in," completing "since I haven't had any flare-ups in," requiring a time duration phrase like "over a year" or "more than a year."


---

## Example idx=345  (MEDIAN band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> …Endoscopy system for length measurement by manual pointing with an electromagnetic tracking sensor
ABSTRACT We propose a practical and simple measurement function of three dimensional (3D) length for endoscopy based on triangulation
using manually pointed correspondences. This system is a novel combination of conventional methods. 3D length information
is useful for many medical purposes and the burden of manual

**GOLD:** <explanation> Medical/technical academic paper format: abstract section establishing problem, method, and utility of an endoscopy measurement system.  Narrative momentum: the abstract is mid-sentence listing clinical motivations, having covered system description and methodology, now detailing practical benefits — "3D length information is useful for many medical purposes" leads naturally to elaborating specific use cases or justifications.  Final token "manual" is part of the phrase "burden of manual," an incomplete noun phrase requiring a noun (e.g., "operation," "measurement," "pointing") describing the effortful human task the system addresses. </explanation>

**raw NLA** (fve +0.632): Academic/technical paper describing a medical imaging system, establishing context for a problem-solution structure. The text is building toward explaining limitations of current methods.

The phrase "manual" strongly signals a contrast with automation — the next word is almost certainly "manual" followed by a noun like "segmentation," "marking," or "labeling," completing the criticism of current methods' inefficiencies.

Final token "manual" is an adjective mid-noun-phrase, requiring an immediate noun or noun phrase describing a labor-intensive task performed by human operators, e.g., "manual segmentation" or "manual marking."

**whitened SFT** (fve +0.178): Academic/technical paper describing a medical imaging system, establishing context for a problem-solution structure — the system's limitations are being enumerated.

The sentence "However, the system has a problem of manual" signals an upcoming description of a limitation or drawback, likely related to manual measurement, calibration, or data entry — a contrast to the system's automated capabilities.

Final token "manual" is an adjective beginning a noun phrase, requiring an immediate noun or noun phrase describing a manual task, e.g., "manual measurement," "manual calibration," or "manual data entry."

**whitened RL** (fve +0.254): Academic paper introducing a novel image analysis system for measuring vascular branching in intraoperative OCT, establishing technical context and clinical motivation throughout.

The paper is building toward explaining the limitations of current manual measurement methods, setting up the contrast between the proposed automated system and the existing labor-intensive process. "However, the measurement requires manual" signals an upcoming contrast clause listing the drawbacks of manual measurement, e.g., "However, the measurement requires manual tracing by a surgeon."

Final token "manual" ends "by manual," completing "requires laborious manual," part of "However, the measurement of the branching points requires manual," immediately followed by "tracing" or "measurement" describing the ted


---

## Example idx=50  (MEDIAN band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> …eeps Westerners believing in the myth of the Western system, making them unable to evaluate non-Western democratic countries, particularly China, in a fair and objective way and face up to flaws within the Western system.
Since the outbreak of the pandemic, China's moves have formed a sharp contrast to the incompetent responses by many Western countries, especially the US. In the face of the health emergency, China has acted promptly and effectively, and it has mobilized the entire society to cooperate and keep solidarity in the epidemic fight. This demonstrates the institutional advantages of

**GOLD:** <explanation> Argumentative momentum: the essay contrasts China's effective pandemic response with Western failures, building toward a conclusion praising China's system.  "End of History" framing: Fukuyama's theory is positioned as a flawed lens distorting Western perception of China, urging fair reassessment.  Final fragment "institutional advantages of" is an incomplete noun phrase mid-sentence, directly requiring a noun or noun phrase naming China's specific system, e.g., "China's system" or "socialism with Chinese characteristics." </explanation>

**raw NLA** (fve +0.693): Political commentary on China's governance model, contrasting Western democracy with China's system, establishing a critical but measured tone throughout.

The sentence structure "the institutional advantages of" sets up a noun phrase completing the thought — specifically naming China's political system, likely "the Chinese socialist system" or "the Chinese model."

Final token "of" is a preposition opening a noun phrase, requiring an immediate noun or noun phrase describing China's institutional/political system, consistent with the preceding discussion of China's governance model.

**whitened SFT** (fve +0.269): Political speech pattern: formal Chinese government rhetoric praising China's pandemic response, listing achievements and institutional strengths.

Narrative momentum: the text is building a case for China's institutional superiority, having just cited specific statistics (100 million, 100,000, 10,000), now pivoting to broader institutional comparison.

Final token "of" opens a prepositional phrase completing "the institutional advantages of [China/the Chinese system]" — a noun phrase naming the specific institution or system being praised is immediately required.

**whitened RL** (fve +0.254): Ongoing Chinese political commentary on the pandemic and governance, establishing that China's system has demonstrated advantages over Western models. The argumentative momentum builds toward explaining how China's collective governance model complements the Communist Party's strengths.

The sentence structure "The advantages of the system of pooling resources to fight the epidemic can be seen as the unique characteristics of" sets up a noun phrase completing "the unique characteristics of," requiring "the Chinese socialist system" or "our country's governance model."

Final token "of" ends "advantages of the system of pooling resources to fight the epidemic of," completing "which reflects the unique characteristics of," requiring "the Chinese socialist system" or "our country's governance


---

## Example idx=586  (MEDIAN band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> …Forgot your password? Enter the email address you signed up with and we will send you a link to create a new password.
Your password reset has been submitted to the address provided.
Standard character mark
Engineering and design of carrier-class communications equipment, including digital transmission, remote monitoring, power distribution and demarcation products used by wireline and wireless telecommunications service providers, industrial customers,

**GOLD:** <explanation> List continuation pattern: the text enumerates product/customer categories with "including digital transmission, remote monitoring, power distribution and demarcation products used by wireline and wireless telecommunications service providers, industrial customers," implying more list items follow.  Domain signal: trademark/patent registration context — "Standard character mark" introduces a formal USPTO-style goods/services description requiring complete, precise enumeration of covered categories.  Final token "customers," is a comma-separated list item mid-enumeration, immediately requiring another customer/entity type (e.g., "government agencies," "utilities," or similar end-user category) to complete the series. </explanation>

**raw NLA** (fve +0.802): Product description genre following technical marketing format, establishing capabilities and use cases for a networking equipment product.

List continuation pattern: "telecom operators, business customers," signals an ongoing enumeration of target market segments, with more industry types expected (e.g., "government agencies," "enterprise customers").

Final token is a comma after "business customers," mid-list, requiring additional parallel noun phrases describing customer types or market sectors to complete the enumeration.

**whitened SFT** (fve +0.247): Technical product description listing applications, following a pattern of enumerating use cases for the optical transponder equipment.

The list of applications is mid-enumeration: "telecom service providers, industrial users," signals more comma-separated entities expected (e.g., "government agencies," "data centers").

Final token is a comma after "industrial users," acting as a list separator, requiring continuation with additional application categories or a closing phrase completing the enumeration.

**whitened RL** (fve +0.254): Technical product description for a fiber optic transmission system, establishing capabilities and applications throughout. The text follows a pattern of listing use cases and customer types.

The sentence structure "used by service providers, utilities," signals an ongoing list of application domains for the product, with "telecom companies, military, and" implying additional industry sectors like "government," "enterprise," or "telecom operators."

Final token "," follows "utilities," completing "for service providers, utilities," as part of a comma-separated list of application domains; next tokens must continue the list with another industry or customer type, e.g., "and industrial applications."


---

## Example idx=390  (WORST band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> …tructure and gas meter equipment management Gas Connections and Infrastructure Our GIRS accredited project managers and designers are the backbone of our gas connections and infrastructure operation. Read More Hydrogen Services We’re embracing the hydrogen revolution. Hydrogen will be an important part of a decarbonated net zero energy system in the future. Read More Energy Managers Effective energy management is a critical component of modern business operations. Read More Construction Industry Accredited by GIRS (Gas Industry Registration Scheme), we are here to provide assistance. Read More

**GOLD:** <explanation> Structured service listing pattern: each entry follows "Title → description → Read More" repetition, establishing strong continuation expectation.  Website services page genre: corporate utility/energy sector tone, bullet-style service summaries with consistent brevity and formal register throughout.  Final token "More" closes a "Read More" call-to-action link, the fourth in a repeating sequence — next tokens likely begin a new service category title or section heading, continuing the established list format. </explanation>

**raw NLA** (fve +0.291): Business services webpage listing company offerings, following a pattern of "Service Category → description" entries. The list structure suggests more service categories may follow.

The page is mid-list, having covered "Gas and Oil," "Water and Sewerage," "Electrical and Communications," and "Construction and Maintenance" — implying additional service categories or a closing section may follow.

"Construction and Maintenance" is the final list item, ending the current section; the next token likely begins a new section header, a closing call-to-action, or a new service category label.

**whitened SFT** (fve -0.843): Marketing/promotional copy for a gas services company, using structured sections with bold headers ("Why Choose Us?", "Our Services") to present company credentials and offerings.

List continuation pattern: "Our Services" section has begun enumerating service categories (Gas Safety, Gas Installation, Gas Maintenance, Gas Metering, Gas Metering), suggesting more service items or a closing summary statement may follow.

Final token "4000" ends a specific statistic about gas meters, part of a promotional claim; next tokens likely continue with additional service details, a call-to-action, or another section header.

**whitened RL** (fve -0.797): Promotional copywriting for a gas services company, establishing credentials and services offered to Australian clients, with a structured listicle format throughout.

The text follows a pattern of listing service benefits and applications, with "Our services include: Gas Leak Detection, Electrical Services, Industrial Solutions, Pipeline Engineering" — more service categories or elaboration are expected to continue this list.

Final token "Refrigeration" ends "Pipeline Engineering" as part of a comma-separated services list ("Gas Leak Detection, Electrical Services, Industrial Solutions, Refrigeration"), immediately expecting another comma and additional service category, or a closing clause completing the list.


---

## Example idx=901  (WORST band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> …rthy, and may not defraud on them with the money or their particular friends.
These females are looking for a man who can create them laugh and feel relaxed around him. They are certainly not afraid expressing their feelings and emotions, so they will be willing to make an effort new things along.
It is vital to remember the particular ladies result from different cultures, they usually have to adjust to their new environment. The best way to earn them over is to be genuine and be happy to show them simply how much you take care of them.
Another thing that will help you to win over these types

**GOLD:** <explanation> Listicle/advice article pattern: text offers sequential tips for attracting European women, with repeated "Another thing" and "It is vital" structures suggesting more advice points follow.  Continuation of "tips for impressing European women" argumentative momentum: the phrase "these types" signals an unfinished noun phrase describing European women, requiring a completion before the next piece of advice.  Final token "types" is mid-noun-phrase, part of "these types of [women/ladies]," immediately requiring a noun or prepositional phrase to complete the reference before continuing the sentence. </explanation>

**raw NLA** (fve -0.370): Spam/SEO content pattern: low-quality, keyword-stuffed dating site promotional text with fragmented sentences and repeated phrases like "free dating sites."

List/enumeration momentum: the text has been cycling through dating site features, benefits, and services, suggesting continuation of similar promotional claims.

Final token "types" ends an incomplete phrase "a number of types," which is part of a list construction requiring continuation — likely "of dating sites" or "of services" followed by another promotional claim about site features.

**whitened SFT** (fve -0.863): Product listing format with bullet-pointed features, establishing a pattern of short, punchy marketing phrases like "No more waiting for your favorite designer to come to you."

The list of product benefits is mid-enumeration, with several items already given; more feature bullets are expected to continue the pattern.

Final token "types" ends an incomplete bullet point ("You can now get a wide range of types"), requiring immediate continuation specifying what types of products/designs are available, likely "of products" or a specific category.

**whitened RL** (fve -0.798): Promotional copy for a beauty/luxury skincare brand, establishing a pattern of listing product features and benefits of online shopping for direct-to-consumer services.

The sentence structure "These include a variety of services such as" sets up an enumeration of specific offerings — likely "of online consultations" or "of a beauty platform" — completing the thought about what the website provides alongside the listed products.

Final token "types" ends "of" completing "for these types," part of "These include a variety of services such as" — next tokens must be a noun phrase specifying the service type, e.g., "of online consultations" or "of a beauty platform."


---

## Example idx=197  (WORST band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> … or CAD Training Certification Program
- Revit Certification
For immediate assistance please send an updated Word version of your resume to Aaron Tibbetts at firstname.lastname@example.org.
Selectek is a member of the American Staffing Association, which promotes legal, ethical, and professional practices for the staffing industry. All employees at Selectek are certified by the American Staffing Association through the Certified Staffing Professional® and Technical Services Certified? programs. ASA certification programs help our employees:
- Protect the rights of our candidates and employees


**GOLD:** <explanation> Job posting genre with bullet-pointed requirements and benefits structure — parallel list items describing organizational/employee benefits expected to continue.  The ASA certification section follows a pattern of explaining member benefits with em-dash bullet points; additional parallel benefits like "Ensure compliance" or "Maintain standards" are strongly anticipated.  Final token "employees" closes the first bullet point in a list of ASA certification program benefits, requiring 1-2 more parallel "- [Verb] the [noun]..." bullet items to complete the promised list. </explanation>

**raw NLA** (fve -0.290): List continuation pattern: a bulleted list of "Benefits of ISO 9001" is mid-sequence, with "Protect employees" as the first item, strongly implying more parallel benefit items follow.

Domain/genre signals: formal business/quality management content, likely from a certification or compliance website, maintaining professional register throughout.

Final token "employees" closes the first list item; the next token must begin another parallel bullet point (e.g., "- Improve customer satisfaction" or "- Enhance business reputation"), continuing the same "- [Benefit]" format.

**whitened SFT** (fve -0.835): Bulleted list structure with parallel formatting: each item follows "- [Topic] [detail]" pattern, establishing strong continuation expectations.

The document is a formal organizational mission statement listing key areas of focus, with "Protecting the rights of employees and consumers" as the first bullet, implying additional parallel mission areas remain.

Final token "employees and consumers" ends the first bullet point; the next token must begin a new "- " bullet continuing the parallel list of organizational mission areas.

**whitened RL** (fve -0.803): Promotional/informational document listing safety and consumer protection services offered by a company, establishing a pattern of describing organizational roles and responsibilities.

The text follows a structured format listing what the company does: "Protecting consumers and employees by enforcing rules and regulations" sets up a continuation of protective measures — likely "Protecting consumers and employees" followed by additional parallel actions like "Monitoring industry standards" or "Promoting safe practices."

Final token "The" opens "Protect people's rights and safety" as part of a list of what the company does; it begins a noun phrase completing "To protect people's rights and safety," requiring a verb phrase or clause describing specific actions taken, e.g., "The company also


---

## Example idx=875  (WORST band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> … good attitude of consultation, finally we achieve a win-win situation,a happy cooperation!
Excy Secjadas Apr 20, 2019 5Customer service staff and sales man are very patience and they all good at English, product's arrival is also very timely, and Oil Field Terms is packaged so well, nothing broke on the road,a good supplier.
Mr. Francois Bertrong Apr 20, 2019 5This is the first business after our company establish, products and services of SHANDONG E.FINE PHARMACY CO., LTD company are very satisfying, we have a good start, we hope to cooperate continuous in the future!
Ahmed Youssef Apr 07, 2

**GOLD:** <explanation> Repeating review pattern: each entry follows "Name | Date | Rating | Review text" format, strongly constraining next tokens to continue Ahmed Youssef's review body.  Positive sentiment momentum: all previous reviews are 5-star with enthusiastic praise for service, quality, packaging, and partnership, predicting similar complimentary content ahead.  The truncated date "Apr 07, 2" is mid-entry (likely "2019"), immediately requiring completion of the date, then a rating number (likely "5"), then review text. </explanation>

**raw NLA** (fve +0.135): Repetition/continuation pattern: The text is a list of customer testimonials/reviews, each following a consistent format of name, location, and praise for a product or service.

Domain/genre signals: E-commerce product review page, likely for a hotel or accommodation service, with positive sentiment throughout.

Final token "2" is the start of a year (e.g., "2023" or "2022"), completing the date "April 2[023]" — the immediate next tokens must be the remaining digits of a year, almost certainly "023" or "022."

**whitened SFT** (fve -0.899): Repetition/continuation pattern: Each entry follows "Quote → Source → Attribution" structure, cycling through multiple quotes about love.

List continuation momentum: The sequence presents a series of love quotes with consistent formatting; another quote, source, or attribution is expected to follow.

Final token "2" is the start of a year (likely "2023" or similar), part of a date stamp or publication year following "Posted on," requiring 3 more digits to complete the year.

**whitened RL** (fve -0.816): Repetition/continuation pattern: The text is a list of Chinese customer reviews for a hotel, each on a new line, with consistent structure: "Review content" + "More reviews."

Semantic completion: The pattern "5 reviews:" followed by four review snippets establishes that a fifth review entry is expected next, continuing the same format.

Final token "2" ends "5 reviews:" as part of "5 reviews:" followed by "2" — likely completing "5 reviews:" as "5 reviews:" + "2" = "5 reviews:" + "2." or "5 reviews:" + "2." continuing the list.


---

## Example idx=90  (WORST band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> …d by a map his best friend in high school made for him, retraces the memories of their friendship before he leaves their hometown for good.
Language: Filipino (Sub: English)
Director: Petersen Vargas
Actor: Ross Pesigan, Earl Policarpio
- THE LIFE OF SILENCE (2016, 26min)
Silence usually comes with a backdrop, waiting to be stirred. A novel and a reality.
Language: Mandarin (Sub: English)
Director: Ying Cheng-ru
- LEE (2011, 6min)
Lee is no ordinary warrior: he has a secret love. When his secret is revealed - Lee must face the hardest battle of her life: to gain respect.
Language: No Dialogue


**GOLD:** <explanation> Structured film festival/catalog format: each entry follows "- TITLE (year, runtime) / description / Language / Director / Actor" pattern consistently.  List continuation momentum: multiple short film entries have been presented; another entry or field completion is expected, with "Actor:" or another metadata tag likely next.  Final token "Dialogue" ends the "Language: No Dialogue" field — the last metadata line of the "LEE" entry — immediately expecting either another metadata field (e.g., "Director:") or a new film entry beginning with "- TITLE". </explanation>

**raw NLA** (fve -0.116): Structured film database format with consistent metadata fields (Title, Director, Writer, Cast, Country, Language, Runtime, Genre, Rating, Description) — next field expected continues this pattern.

The list of metadata fields follows a strict alphabetical/sequential order, with "No Dialogue" as the final entry suggesting the next field is a new metadata category (e.g., "Noir," "Thriller," or "Starring").

"No Dialogue" is the last metadata field value, likely the final item in a structured list; the next token is almost certainly a new metadata label (e.g., "Noir," "Thriller," or "Starring") continuing the film's descriptive attributes.

**whitened SFT** (fve -0.670): Structured film metadata format: each line follows "Label: Value" pattern (Title, Director, Writer, etc.), establishing strong continuation expectations.

The list of metadata fields is incomplete — typical film database entries include additional fields like "Cast," "Release Date," "Runtime," "Genre," or "Country."

Final token "None" closes the "Language" field value; the next token is almost certainly a newline followed by another metadata label (e.g., "Cast:" or "Release Date:").

**whitened RL** (fve -0.825): Ongoing analysis of "The Last Kingdom" Season 2's reception and production context, establishing that the show's lack of dialogue in the finale episode is being contrasted with other non-verbal storytelling techniques.

The sentence structure "Despite having no dialogue, the episode was well-received" sets up an explanation of why non-verbal storytelling works effectively — likely referencing visual cues, symbolism, or the show's established visual style.

Final token "No dialogue" ends "for a show with No dialogue" — completing "Despite having No dialogue" clause, immediately requiring continuation explaining the show's visual storytelling or production context, e.g., "Director" or "Director's approach."


---

## Example idx=501  (WORST band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> …e gambling authorities, All Pokies Mobile Casino also offers a variety of table games. Once your withdrawal has been approved, if you receive a $100 bonus with a 30x wagering requirement. The casino boasts over 500 different pokies games, you would need to wager $3,000 (30 x $100) before you could cash out any winnings.
These developers use state-of-the-art technology to create games that are visually stunning, enhance the betting experience. Free demo play online pokies are also a great way for players to practice their skills, simple gameplay. In this article, and potential for big payouts.


**GOLD:** <explanation> Listicle/promotional casino content pattern: fragmented clauses listing features like "visually stunning, enhance the betting experience, simple gameplay" suggest continuation with more comma-separated selling points or a new promotional paragraph.  Topic cycling through pokies/casino features: text repeatedly shifts between bonuses, mobile play, game variety, and wagering requirements, suggesting next tokens introduce another casino feature or tip.  Final token "payouts." ends a sentence fragment list; a new sentence or paragraph header continuing the promotional Australian pokies/casino theme is immediately expected. </explanation>

**raw NLA** (fve +0.194): Listicle/promotional casino content pattern: each paragraph introduces a new slot game or feature, suggesting continuation of this format.

The text follows a structured review format covering casino features, game types, and player benefits, implying further elaboration on specific games or features.

Final sentence ends with "slot machines" as a topic pivot, likely introducing a new paragraph or section discussing slot machine specifics, game types, or player tips — a new sentence or paragraph beginning is immediately expected.

**whitened SFT** (fve -0.865): Structured listicle format with numbered sections (1, 2, 3) signals continuation of a multi-part guide, with section 3's content now expected to follow.

The text follows a consistent pattern: bolded section header, then explanatory paragraph(s) covering benefits, considerations, and practical advice — section 3's body content is now due.

The final sentence ends mid-thought after "in the short term," completing a contrast between short-term and long-term benefits; the next tokens will likely elaborate on long-term benefits or transition to actionable advice for section 3.

**whitened RL** (fve -0.881): Business proposal document listing services and benefits for a local casino, establishing a pattern of structured sections with headers and descriptive content about community engagement and partnership.

The text has detailed a multi-step process for selecting and implementing services, with "Next Steps" section already covering initial assessment and community input — the next logical continuation would elaborate on additional steps like onboarding, training, or specific service recommendations.

Final token "." ends "of the state." completing "For instance, we've identified various activities to enhance engagement, including slot tournaments and community events." This clause closes a list of examples, so the next tokens likely begin a new sentence continuing the "Next Steps" elaboratio


---

## Example idx=684  (WORST band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> …g Plant Capacity 50 TPH 70 Lakh Vindhya Engineering Shalimar Engineering Impact Crusher Quarry Plant And Stone Crushing For Coal Industry 25 Lakh Shalimar Engineering Stainless Steel Mobile Crushing Plant 12 Lakh Get Latest Price Capacity Customize.Aug 24 2003 The stone crushing plant employs many people both skilled and unskilled who are involved in the entire process of mining crushing and transportation of the crushed stone The plant’s ideal location is supposed to be near growing towns or cities so as to supply the demand for the stone and at the same time near the source to the.Read More


**GOLD:** <explanation> Domain/genre signals: SEO-aggregated mining/crushing equipment content, mixing product listings, regulatory snippets, and operational guides with repetitive "Read More" pagination markers throughout.  Repetition/continuation pattern: Each section follows a heading → descriptive content → "Read More" structure, cycling through crushing plant subtopics like operations, maintenance, design, and equipment specifications.  Final token context: The text ends mid-sentence ("near the source to the"), part of a location-rationale clause about stone crushing plant siting; the immediate next tokens must complete this prepositional phrase, likely "raw materials" or similar, followed by a period and "Read More." </explanation>

**raw NLA** (fve -0.619): Repetitive SEO/spam content pattern: each paragraph introduces a new "gold ore crusher" topic with near-identical phrasing, suggesting another similar entry follows.

List continuation momentum: the text cycles through gold ore crusher topics (types, manufacturers, prices, equipment), implying another such topic entry is expected next.

Final token "Gold" opens a new sentence/paragraph, likely beginning a new topic entry (e.g., "Gold ore crusher plant," "Gold ore processing," or "Gold ore crusher machine"), constrained to continue the established SEO keyword-stuffed format.

**whitened SFT** (fve -0.875): Domain/genre signals: Industrial mining equipment catalog/SEO content, mixing product listings, technical specs, and fragmented web-scraped text about crushing and grinding machinery.

Repetition/continuation patterns: The text repeatedly cycles through equipment types (jaw crusher, cone crusher, ball mill, grinding mill), manufacturers, and specifications, suggesting more of the same fragmented product descriptions follow.

Final token: "Crushing" ends a sentence fragment "The main objective of the crushing process is to reduce the size of the material to a certain size" — a technical definition sentence, likely followed by a period or continuation explaining further crushing process details or transitioning to another equipment description.

**whitened RL** (fve -0.896): Technical document listing crusher equipment specifications and applications, establishing a pattern of detailed product descriptions with model names, capacities, and process details.

The text follows a structured enumeration pattern: "Here are some examples of the crusher equipment used in the process of stone crushing in China" signals that more specific equipment details or process steps will follow, continuing the listicle format.

Final token "Equipment" ends "Stone Crushing Process Equipment" as part of "Details about the Crusher Equipment" clause, completing "Here are some details about the Crusher Equipment" — next tokens must continue the sentence, likely "for Stone Crushing Process" or "used in the Stone Crushing Process."


---

## Example idx=285  (WORST band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> …makers have avoided changes to the big federal health and retirement programs, the debt would start rising again after 2023 as the bulk of the baby-boom generation retires.
It would be reassuring if the moderate Democrats who assured us Obama was really concerned about a fiscal train wreck would denounce his irresponsibility and/or concede they were conned.
But the president will have a bigger problem than Republicans to worry about if he ignores the massive debt: rating firms and the bond market. In 2011 an actual deal on the debt (which gave birth to the sequestration now in contention) didn

**GOLD:** <explanation> Political opinion journalism arguing Obama's debt position is irresponsible, using bipartisan sources and CBO data as evidence; narrative builds toward consequences of inaction.  The argument has escalated from political criticism to economic market consequences, introducing "rating firms and bond market" as a new, serious threat beyond partisan politics.  Final clause "In 2011 an actual deal on the debt (which gave birth to the sequestration now in contention) didn" — mid-sentence, past-tense narrative about 2011 debt deal, with "didn" requiring completion of a negative verb phrase, likely "didn't prevent" or "didn't satisfy" rating agencies/S&P downgrade. </explanation>

**raw NLA** (fve -0.648): News article reporting style with factual, journalistic tone — expects continuation of factual details about the incident.

Narrative momentum: the article has established the event (date, location, victim, cause) and is now detailing the specific circumstances of the crash, following a standard news structure.

Final token "the" is an article beginning a noun phrase, likely introducing a specific object or location related to the crash (e.g., "the car," "the road," "the intersection") — a noun phrase must immediately follow.

**whitened SFT** (fve -0.942): Product listing format with bullet-pointed feature descriptions, establishing a pattern of short, punchy marketing claims about a gaming mouse.

The list of bullet points has been building a feature-by-feature breakdown; the pattern suggests another feature or closing detail is expected next.

The final token "2" begins a new bullet point, likely continuing a numeric feature (e.g., "2.4GHz," "2000 DPI," or "2-year warranty"), constrained to a short, capitalized product specification phrase.

**whitened RL** (fve -0.896): Product listing format for a Chinese e-commerce item, establishing technical specifications and features of a 3D printer. The listing follows a structured pattern of describing capabilities, materials, and compatibility.

The listing has enumerated several technical specifications and features, with "The" beginning a new clause continuing the description of the printer's capabilities or additional features — likely "The printer supports..." or "The software is compatible with..."

Final token "The" opens a noun phrase completing "In addition, the," continuing the list of product attributes; it must be followed by a noun phrase describing another feature, capability, or specification of the printer.


---

## Example idx=918  (WORST band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> …as just a tad sweet and didn’t necessarily meet the flavor profile of what most pipemen would consider a Balkan, but it was very good none the less, so much so that I will buy it and add it to my everyday rotation, as I surely will CD Blend and Maple Street. Altadis was extremely generous with their samples, so I have quite a number of their other blends to try. At some point in the not to distant future, I will review some of these in detail, but for the present, I felt moved to alert the readership of this fine electronic publication to the Sutliff series from Altadis. My hat is off to Altad

**GOLD:** <explanation> Enthusiastic review of Altadis/Sutliff pipe tobacco building toward a strong closing endorsement, with narrative momentum toward a positive conclusion praising the company.  The author is mid-sentence completing "My hat is off to Altad-" — a conventional complimentary idiom expressing respect, requiring completion of the company name "Altadis" followed by a closing sentiment congratulating or thanking them for improving their product line.  Final token "Altad" is a truncated proper noun ("Altadis"), syntactically and semantically requiring completion of that name, then likely a period or brief closing praise. </explanation>

**raw NLA** (fve -0.453): Ongoing list of "most popular" articles, establishing a pattern of article titles with dates and brief descriptions, continuing the "most popular" series.

The text is mid-sentence describing the second article's content, following the same structure as the first article's description ("This article discusses...").

The final token "the" is an article beginning a noun phrase, requiring an immediate noun or adjective+noun describing a specific topic or entity related to the second article's content (e.g., "the importance of," "the role of," "the impact of").

**whitened SFT** (fve -0.990): Product listing format with bullet-pointed feature descriptions, establishing a pattern of short, punchy marketing claims about a gaming mouse.

The list of bullet points has been building a feature-by-feature breakdown of the mouse's specs (sensitivity, buttons, weight, cable), suggesting more technical specifications or selling points remain.

The final token "2" begins a new bullet point, likely continuing a numeric specification (e.g., "2.4GHz wireless," "2000 DPI," or "2-year warranty"), constrained to a short technical or marketing phrase starting with a number.

**whitened RL** (fve -0.928): Product listing format for a Chinese e-commerce item, establishing technical specifications and features of a 3D printer. The listing follows a structured pattern of describing capabilities, materials, and compatibility.

The listing has enumerated several technical specifications and features, with "The" beginning a new clause continuing the description of the printer's capabilities or additional features — likely "The printer supports..." or "The software is compatible with..."

Final token "The" opens a noun phrase completing "In addition, the," continuing the list of product attributes; it must be followed by a noun phrase describing another feature, capability, or specification of the printer.


---

## Example idx=181  (WORST band for whitened-RL)

**Source text (activation = layer-24 state at its END):**

> …at not all of the retailers in private equity portfolios are ready for the prime time of public markets, said Joe Pellegrini, managing director of Robert W. Baird & Co.’s consumer retail team.
“What investors value is growth, number one, but predictable growth is even more critical,” Pellegrini said. That puts a retailer with 100 stores, comparable-store sales growth and a chance to take advantage of the weak real estate market and open more doors at the top of many people’s lists.
So, could all this potential activity add up to a return of fashion M&A where prices go higher and higher?
“I don

**GOLD:** <explanation> Financial journalism article analyzing M&A activity in fashion/retail sector, establishing an expert-driven, analytical tone throughout with executive quotes.  Narrative momentum of "question posed, expert answer expected": the article consistently follows a pattern of raising a question or topic, then providing a named expert quote as response.  Final token "don" is the start of a direct quote from a named source (following the pattern "'I don[']t...'" or similar), part of an answering statement to the closing rhetorical question about whether fashion M&A prices will escalate — the next tokens will almost certainly continue this skeptical or cautionary expert response. </explanation>

**raw NLA** (fve -0.672): News article reporting style with factual, journalistic tone — expects continued factual elaboration of the incident details.

Narrative momentum: the article has introduced the incident, quoted officials, and is now detailing the specific circumstances of the crash, with more technical/operational details expected.

Final token "the" opens a noun phrase mid-sentence, requiring an immediate noun or adjective+noun describing what the plane was doing or what happened next (e.g., "the plane was flying," "the pilot reported," "the aircraft was").

**whitened SFT** (fve -1.024): Numbered list pattern: items 1–4 follow a consistent structure (number, title, description), establishing strong continuation expectation for item 5.

Each entry follows a template: title, then 1-2 sentence description of a specific historical/educational resource, maintaining an informational, encyclopedic tone throughout.

The final token "5" is the start of the fifth list item, immediately requiring a period or dash followed by a new resource title (likely another historical document, law, or educational tool).

**whitened RL** (fve -0.966): Ongoing listicle format: "Here are some of the most popular Chinese dating apps" signals an enumeration of app names and descriptions, with "Tantan" already introduced as the first example.

The sentence structure "The app is popular because it allows users to..." establishes a pattern of explaining app features/benefits, so next tokens will continue describing Tantan's specific features or capabilities.

Final token "": ends "2" completing "201" as part of "201" — likely "2019" or "2018," completing "In recent years, the app has grown rapidly" clause, requiring a year number immediately after.
