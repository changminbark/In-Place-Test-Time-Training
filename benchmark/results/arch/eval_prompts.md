# Eval prompt templates per task

Each section shows the rendered prompt that would be passed to `model.generate()` for one sampled example, in each mode. Long bodies are truncated with a head/tail view; the elision marker shows how many characters are skipped.

## `vt`  @  1024 tokens

- **Gold answer**: `'OHWXG'`
- **Aliases** (5): `['OHWXG', 'IRQOP', 'NAIVY']` …
- **Prompt provenance**: HELMET-rendered (`example.prompt` field)

### in_context / ttt_paper prompt

```
Memorize and track the chain(s) of variable assignment hidden in the following text.

The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
VAR UMP = 57176
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
VAR KLS = VAR UMP 
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
VAR DJK = VAR KLS 
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
VAR BIG = VAR DJK 
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass i

... [2,125 characters elided] ...

R JNABE = VAR LXQOE 
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
Question: Find all variables that are assigned the value 38754 in the text above. Answer: According to the chain(s) of variable assignment in the text above, 5 variables are assigned the value 38754, they are:
```

### ttt_strict — phase 1 (ingest, document only)

```
Memorize and track the chain(s) of variable assignment hidden in the following text.

The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
VAR UMP = 57176
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
VAR KLS = VAR UMP 
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
VAR DJK = VAR KLS 
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
VAR BIG = VAR DJK 
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass i

... [1,915 characters elided] ...

e go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
VAR JNABE = VAR LXQOE 
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
```

### ttt_strict — phase 2 (answer, question only)

```
Question: Find all variables that are assigned the value 38754 in the text above. Answer: According to the chain(s) of variable assignment in the text above, 5 variables are assigned the value 38754, they are:
```

## `cwe`  @  1024 tokens

- **Gold answer**: `'turn'`
- **Aliases** (10): `['turn', 'jumper', 'amendment']` …
- **Prompt provenance**: HELMET-rendered (`example.prompt` field)

### in_context / ttt_paper prompt

```
Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.
1. epee 2. lackadaisical 3. brewer 4. syndrome 5. lackadaisical 6. syndrome 7. syndrome 8. theory 9. poem 10. poem 11. digestion 12. support 13. support 14. rehabilitate 15. playwright 16. theory 17. poem 18. lava 19. support 20. lava 21. grasp 22. cayenne 23. rehabilitate 24. overclocking 25. overclocking 26. digestion 27. kindhearted 28. sash 29. lava 30. grasp 31. grasp 32. theory 33. provision 34. lackadaisical 35. digestion 36. rehabilitate 37. programme 38. overclocking 39. tease 40. jeweller
Question: What are the 10 most common words in the above list? Answer: The top 10 words that appear most often in the list are: 1. rehabilitate 2. digestion 3. grasp 4

... [1,083 characters elided] ...

shipper 74. visit 75. visit 76. amendment 77. stripe 78. acted 79. banquette 80. obnoxious 81. callous 82. shipper 83. jumper 84. shutdown 85. obnoxious 86. hippodrome 87. climb 88. craw 89. catalogue 90. turn 91. shut 92. astrakhan 93. fledgling 94. shut 95. amendment 96. jumper 97. board 98. execution 99. thank 100. catalogue 101. vagrant 102. periodical 103. quantity
Question: What are the 10 most common words in the above list? Answer: The top 10 words that appear most often in the list are:
```

### ttt_strict — phase 1 (ingest, document only)

```
Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.
1. epee 2. lackadaisical 3. brewer 4. syndrome 5. lackadaisical 6. syndrome 7. syndrome 8. theory 9. poem 10. poem 11. digestion 12. support 13. support 14. rehabilitate 15. playwright 16. theory 17. poem 18. lava 19. support 20. lava 21. grasp 22. cayenne 23. rehabilitate 24. overclocking 25. overclocking 26. digestion 27. kindhearted 28. sash 29. lava 30. grasp 31. grasp 32. theory 33. provision 34. lackadaisical 35. digestion 36. rehabilitate 37. programme 38. overclocking 39. tease 40. jeweller
Question: What are the 10 most common words in the above list? Answer: The top 10 words that appear most often in the list are: 1. rehabilitate 2. digestion 3. grasp 4

... [955 characters elided] ...

rival 63. attorney 64. shut 65. visit 66. rambunctious 67. visit 68. hippodrome 69. absence 70. turn 71. savings 72. jumper 73. shipper 74. visit 75. visit 76. amendment 77. stripe 78. acted 79. banquette 80. obnoxious 81. callous 82. shipper 83. jumper 84. shutdown 85. obnoxious 86. hippodrome 87. climb 88. craw 89. catalogue 90. turn 91. shut 92. astrakhan 93. fledgling 94. shut 95. amendment 96. jumper 97. board 98. execution 99. thank 100. catalogue 101. vagrant 102. periodical 103. quantity
```

### ttt_strict — phase 2 (answer, question only)

```
Question: What are the 10 most common words in the above list? Answer: The top 10 words that appear most often in the list are:
```

## `fwe`  @  1024 tokens

- **Gold answer**: `'bohdmc'`
- **Aliases** (3): `['bohdmc', 'cymvxz', 'ltyikj']` …
- **Prompt provenance**: HELMET-rendered (`example.prompt` field)

### in_context / ttt_paper prompt

```
Read the following coded text and track the frequency of each coded word. Find the three most frequently appeared coded words. ... ... ... ltyikj ... cymvxz ... ... ... ... ... ... ... ... ... ... cymvxz ltyikj ... ... ... hmsjih bohdmc ... ... ... ... ... ... cymvxz ... ... bohdmc cymvxz ... ... ... bohdmc ... ... ... ... ... aypflr ... ... ... ... hmsjih cymvxz hmsjih htksla bohdmc bohdmc ... ... bohdmc ... ... ... ... ... ... ... ... cymvxz ... ... ... ... ... ... ... ... ... ... hmsjih cymvxz ... ... ... ... cymvxz ... ... bohdmc ... ... bohdmc ltyikj ... bohdmc ... ... ... bohdmc ... ltyikj ... ... cymvxz bohdmc bohdmc ... ... ... ... bohdmc ... cymvxz ... ... ... ... ... htksla ... cymvxz ltyikj htksla ltyikj ... ltyikj aypflr ... wmtbbq ... cymvxz ... ... ... hmsjih ... ... htksla .

... [1,282 characters elided] ...

ohdmc ... bohdmc cymvxz bohdmc ... ... ... ... ... bohdmc ... ... bohdmc ... cymvxz hmsjih ... ... bohdmc cymvxz ... ... ... zrrdur ... ... ... bohdmc ltyikj ... cymvxz ... ... ... hmsjih bohdmc ... ... zrrdur ltyikj ... cymvxz bohdmc bohdmc ... cymvxz ... bohdmc
Question: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text? Answer: According to the coded text above, the three most frequently appeared words are:
```

### ttt_strict — phase 1 (ingest, document only)

```
Read the following coded text and track the frequency of each coded word. Find the three most frequently appeared coded words. ... ... ... ltyikj ... cymvxz ... ... ... ... ... ... ... ... ... ... cymvxz ltyikj ... ... ... hmsjih bohdmc ... ... ... ... ... ... cymvxz ... ... bohdmc cymvxz ... ... ... bohdmc ... ... ... ... ... aypflr ... ... ... ... hmsjih cymvxz hmsjih htksla bohdmc bohdmc ... ... bohdmc ... ... ... ... ... ... ... ... cymvxz ... ... ... ... ... ... ... ... ... ... hmsjih cymvxz ... ... ... ... cymvxz ... ... bohdmc ... ... bohdmc ltyikj ... bohdmc ... ... ... bohdmc ... ltyikj ... ... cymvxz bohdmc bohdmc ... ... ... ... bohdmc ... cymvxz ... ... ... ... ... htksla ... cymvxz ltyikj htksla ltyikj ... ltyikj aypflr ... wmtbbq ... cymvxz ... ... ... hmsjih ... ... htksla .

... [1,045 characters elided] ...

. bohdmc ltyikj ... ... bohdmc ... zrrdur bohdmc bohdmc ... bohdmc ... wtmlse ... dhjavx bohdmc ... ltyikj bohdmc ... cymvxz ... ... cymvxz ... htksla ... bohdmc ... bohdmc bohdmc ... ... ... bohdmc nxexvk ... bohdmc ... bohdmc ... ... bohdmc ... bohdmc cymvxz bohdmc ... ... ... ... ... bohdmc ... ... bohdmc ... cymvxz hmsjih ... ... bohdmc cymvxz ... ... ... zrrdur ... ... ... bohdmc ltyikj ... cymvxz ... ... ... hmsjih bohdmc ... ... zrrdur ltyikj ... cymvxz bohdmc bohdmc ... cymvxz ... bohdmc
```

### ttt_strict — phase 2 (answer, question only)

```
Question: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text? Answer: According to the coded text above, the three most frequently appeared words are:
```

## `helmet_trec_coarse`  @  8192 tokens

- **Gold answer**: `'3'`
- **Prompt provenance**: HELMET-rendered (`example.prompt` field)

### in_context / ttt_paper prompt

```
Use the provided mapping from the text to label to assign a label to the text. Only output "label: {label}" and nothing else. 

What is a film starring Jude Law ?
label: 4

How many U.S. presidents were assassinated during Queen Victoria 's reign ?
label: 0

What is Plc ?
label: 3

What 's the abbreviation for limited partnership ?
label: 1

Who was the president of Vichy France ?
label: 2

What mountain range is traversed by the highest railroad in the world ?
label: 5

What U.S. state is Dixville Notch in ?
label: 5

What does e.g. stand for ?
label: 1

Who was the first woman governor of Wyoming ?
label: 2

What do you call a Poker hand with five cards of the same suit ?
label: 4

How successful is arometherapy ?
label: 3

How tall is the Matterhorn ?
label: 0

What country contains the

... [22,934 characters elided] ...

or James J. Kilroy designate equipment as being satisfactory ?
label: 3

Who is always trying to get the rent from Andy Capp ?
label: 2

What nationality is Ileana Cotrubas ?
label: 5

Where can I find a case on Americans with Disabilities Act of 199 ?
label: 5

What President once told Gene Autry : `` Please give my regards to your wife Dale '' ?
label: 2

What is Doegs ?
label: 3

What is the abbreviated expression for the National Bureau of Investigation ?
label: 1

What are polymers ?
label:
```

### ttt_strict — phase 1 (ingest, document only)

```
What is a film starring Jude Law ?
label: 4

How many U.S. presidents were assassinated during Queen Victoria 's reign ?
label: 0

What is Plc ?
label: 3

What 's the abbreviation for limited partnership ?
label: 1

Who was the president of Vichy France ?
label: 2

What mountain range is traversed by the highest railroad in the world ?
label: 5

What U.S. state is Dixville Notch in ?
label: 5

What does e.g. stand for ?
label: 1

Who was the first woman governor of Wyoming ?
label: 2

What do you call a Poker hand with five cards of the same suit ?
label: 4

How successful is arometherapy ?
label: 3

How tall is the Matterhorn ?
label: 0

What country contains the westernmost point in South America ?
label: 5

How many innings constitute an official baseball game ?
label: 0

What does Larr

... [22,778 characters elided] ...

 4

How did shipyard inspector James J. Kilroy designate equipment as being satisfactory ?
label: 3

Who is always trying to get the rent from Andy Capp ?
label: 2

What nationality is Ileana Cotrubas ?
label: 5

Where can I find a case on Americans with Disabilities Act of 199 ?
label: 5

What President once told Gene Autry : `` Please give my regards to your wife Dale '' ?
label: 2

What is Doegs ?
label: 3

What is the abbreviated expression for the National Bureau of Investigation ?
label: 1
```

### ttt_strict — phase 2 (answer, question only)

```
Answer the following question with just the value, no explanation.

Question: What are polymers ?
label:
Answer:
```

## `helmet_banking77`  @  8192 tokens

- **Gold answer**: `'0'`
- **Prompt provenance**: HELMET-rendered (`example.prompt` field)

### in_context / ttt_paper prompt

```
Use the provided mapping from the text to label to assign a label to the text. Only output "label: {label}" and nothing else. 

Since when do you charge to make a withdrawal? I've always done it for free. So how much is it now?
label: 27

I made a card payment, but it has been declined?
label: 58

Why hasn't the app verified my identity?
label: 74

help, I see money missing I didn't take out.
label: 44

Why was a fee charged to me for transferring money?
label: 39

I cannot get to my app, what should I do?
label: 52

My refund isn't showing up on my statement.
label: 29

I would like to pay by cheque.
label: 70

I did a transfer but it is still pending.
label: 8

Can you tell me what countries you offer support for?
label: 51

How long should it take for my top-up to finish? I've been wait

... [22,855 characters elided] ...

verted. Please look into this issue.
label: 69

Would it be possible to open up an account for children?
label: 7

I need a new card and I live in the United States.
label: 51

there is a debit i dont recognize
label: 32

My account was charged twice for one purchase
label: 23

what is the cost for exchanging currencies?
label: 16

Can I withdraw from any ATM?
label: 9

why has a cash withdrawal charged me?
label: 27

what is source of my money
label: 38

I would like to close my account.
label:
```

### ttt_strict — phase 1 (ingest, document only)

```
Since when do you charge to make a withdrawal? I've always done it for free. So how much is it now?
label: 27

I made a card payment, but it has been declined?
label: 58

Why hasn't the app verified my identity?
label: 74

help, I see money missing I didn't take out.
label: 44

Why was a fee charged to me for transferring money?
label: 39

I cannot get to my app, what should I do?
label: 52

My refund isn't showing up on my statement.
label: 29

I would like to pay by cheque.
label: 70

I did a transfer but it is still pending.
label: 8

Can you tell me what countries you offer support for?
label: 51

How long should it take for my top-up to finish? I've been waiting a while.
label: 11

How do I know it is a Mastercard ATM?
label: 9

This is terrible, i want to delete my account
label: 0



... [22,685 characters elided] ...

, I have done a topup, but my money was reverted. Please look into this issue.
label: 69

Would it be possible to open up an account for children?
label: 7

I need a new card and I live in the United States.
label: 51

there is a debit i dont recognize
label: 32

My account was charged twice for one purchase
label: 23

what is the cost for exchanging currencies?
label: 16

Can I withdraw from any ATM?
label: 9

why has a cash withdrawal charged me?
label: 27

what is source of my money
label: 38
```

### ttt_strict — phase 2 (answer, question only)

```
Answer the following question with just the value, no explanation.

Question: I would like to close my account.
label:
Answer:
```
