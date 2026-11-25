# Confidence of the map's answer (map_confidence)

Made by `src/recommendation/build_map_confidence.py` on 2026-11-25 08:24 UTC (code commit `ce6951b542`), from the new-fields test of the model on the map (RF[all_sat] tuned, `reports/tuning_v1`). Decided on 25 Nov 2026 after looking at the map: see `docs/training_protocol.md`, section 9.

**Built-in check: passed.** The five refitted fold models give exactly the tuning_v1 share of first crops right in every district: Musanze 0.750, Nyabihu 0.744, Nyagatare 0.759, Ruhango 0.629.

## How the map answers

1. Season. Only season B was tested, so another season gets no answer.
2. Distance. The field inputs come from the clicked 2 km block or the nearest CIAT block in the district. Beyond 20 km from any surveyed field there is no answer.
3. One crop or two. If the model's first two crops are 0.10 apart or more, the answer is one crop and its confidence is the share of tested fields like it (same district, same distance band, not close) where the first crop was the crop planted. If they are closer, the answer is the two crops and its confidence is the share of close calls in that district and band where the crop planted was one of the two. A one-crop rate below 50 percent turns into a two-crop answer with the two-crop rate of all fields.
4. Level. High from 75 percent, medium from 50 percent, as before. Below 50 percent the map gives no answer instead of a low one, so a farmer never gets one crop with a low confidence.

## The table

Share right, of how many tested fields. Within 5 km: the fields' own inputs (the tuning_v1 test). 5 to 10 km and 10 to 20 km: the same fields predicted with the inputs of a training block that far away in their own district. A cell with fewer than 30 fields takes the value of the four districts.

| field data from   | answer           | Musanze                     | Nyabihu                     | Nyagatare                   | Ruhango    |
|:------------------|:-----------------|:----------------------------|:----------------------------|:----------------------------|:-----------|
| within 5 km       | one crop         | 76% of 153                  | 78% of 147                  | 78% of 1031                 | 65% of 427 |
| within 5 km       | two crops, close | 87% of 146 (four districts) | 87% of 146 (four districts) | 98% of 53                   | 79% of 61  |
| within 5 km       | two crops, any   | 89% of 164                  | 97% of 168                  | 97% of 1084                 | 88% of 488 |
| 5 to 10 km        | one crop         | 58% of 159                  | 72% of 166                  | 73% of 1055                 | 56% of 409 |
| 5 to 10 km        | two crops, close | 91% of 115 (four districts) | 91% of 115 (four districts) | 91% of 115 (four districts) | 92% of 79  |
| 5 to 10 km        | two crops, any   | 77% of 164                  | 96% of 168                  | 94% of 1084                 | 90% of 488 |
| 10 to 20 km       | one crop         | 32% of 164                  | 45% of 144                  | 72% of 1061                 | 55% of 420 |
| 10 to 20 km       | two crops, close | 85% of 115 (four districts) | 85% of 115 (four districts) | 85% of 115 (four districts) | 99% of 68  |
| 10 to 20 km       | two crops, any   | 73% of 164                  | 70% of 168                  | 94% of 1084                 | 86% of 488 |

## Replay on the tested fields (season B, within 5 km)

What each of the 1,904 tested fields would have been shown, and how often it was right (first crop for a one-crop answer, one of the two for a two-crop answer).

| shown            | until 25 Nov             | 25 Nov, first crop right   | now, by situation        |
|:-----------------|:-------------------------|:---------------------------|:-------------------------|
| one crop, high   | 62% of fields, right 77% | 62% of fields, right 77%   | 70% of fields, right 77% |
| one crop, low    | 8% of fields, right 47%  | 4% of fields, right 48%    |                          |
| one crop, medium | 30% of fields, right 66% | 34% of fields, right 66%   | 22% of fields, right 65% |
| two crops, high  |                          |                            | 8% of fields, right 87%  |

## Replay on the map (season B, a point about every 1 km)

2,943 points. Before: the rule of the morning of 25 Nov (district hit rate, one step lower for a close call or data more than 5 km away). Now: this table.

| rule   | shown on the map   | Musanze   | Nyabihu   | Nyagatare   | Ruhango   |
|:-------|:-------------------|:----------|:----------|:------------|:----------|
| before | one crop, high     | 93%       |           | 77%         |           |
| before | one crop, low      |           | 13%       | under 1%    | 19%       |
| before | one crop, medium   | 7%        | 87%       | 23%         | 81%       |
| now    | no answer          |           |           | under 1%    |           |
| now    | one crop, high     | 92%       | 87%       | 77%         |           |
| now    | one crop, medium   | 6%        | 10%       | 19%         | 85%       |
| now    | two crops, high    | 1%        | 3%        | 3%          | 15%       |

## How to read this

- These are tested rates, not promises: about 1 in 4 high-confidence answers was still wrong in testing.
- A two-crop answer means both crops fit the field about equally. The farmer can choose between them, for example by the seed and market they have.
- The rates within 5 km are conservative for a point inside a surveyed block: the model on the map has seen the block's fields, the tested models had not.
