# Confidence of the map's answer (map_confidence)

Made by `src/recommendation/build_map_confidence.py` on 2026-11-25 10:01 UTC (code commit `2f4b559305`), from the new-fields test of the model on the map (RF[field_sat] steady+weights, `reports/tuning_v3`). Decided on 25 Nov 2026 after looking at the map: see `docs/training_protocol.md`, section 9.

**Built-in check: passed.** The five refitted fold models give exactly the `tuning_v3` share of first crops right in every district: Musanze 0.756, Nyabihu 0.720, Nyagatare 0.756, Ruhango 0.637.

## How the map answers

1. Season. Only season B was tested, so another season gets no answer.
2. Distance. The field inputs come from the clicked 2 km block or the nearest CIAT block in the district. Beyond 20 km from any surveyed field there is no answer.
3. One crop or two. If the model's first two crops are 0.10 apart or more, the answer is one crop and its confidence is the share of tested fields like it (same district, same distance band, not close) where the first crop was the crop planted. If they are closer, the answer is the two crops and its confidence is the share of close calls in that district and band where the crop planted was one of the two. A one-crop rate below 50 percent turns into a two-crop answer with the two-crop rate of all fields.
4. Level. High from 75 percent, medium from 50 percent, as before. Below 50 percent the map gives no answer instead of a low one, so a farmer never gets one crop with a low confidence.

## The table

Share right, of how many tested fields. Within 5 km: the fields' own inputs (the `tuning_v3` test). 5 to 10 km and 10 to 20 km: the same fields predicted with the inputs of a training block that far away in their own district. A cell with fewer than 30 fields takes the value of the four districts.

| field data from   | answer           | Musanze                     | Nyabihu                     | Nyagatare                  | Ruhango                    |
|:------------------|:-----------------|:----------------------------|:----------------------------|:---------------------------|:---------------------------|
| within 5 km       | one crop         | 77% of 151                  | 75% of 154                  | 77% of 1017                | 64% of 428                 |
| within 5 km       | two crops, close | 94% of 154 (four districts) | 94% of 154 (four districts) | 94% of 67                  | 93% of 60                  |
| within 5 km       | two crops, any   | 87% of 164                  | 94% of 168                  | 96% of 1084                | 90% of 488                 |
| 5 to 10 km        | one crop         | 55% of 150                  | 71% of 164                  | 72% of 1072                | 57% of 463                 |
| 5 to 10 km        | two crops, close | 89% of 55 (four districts)  | 89% of 55 (four districts)  | 89% of 55 (four districts) | 89% of 55 (four districts) |
| 5 to 10 km        | two crops, any   | 76% of 164                  | 95% of 168                  | 94% of 1084                | 88% of 488                 |
| 10 to 20 km       | one crop         | 30% of 159                  | 42% of 154                  | 72% of 1068                | 56% of 440                 |
| 10 to 20 km       | two crops, close | 58% of 83 (four districts)  | 58% of 83 (four districts)  | 58% of 83 (four districts) | 46% of 48                  |
| 10 to 20 km       | two crops, any   | 63% of 164                  | 66% of 168                  | 91% of 1084                | 84% of 488                 |

## Replay on the tested fields (season B, within 5 km)

The three rules applied to the predictions of the model on the map now. What each of the 1,904 tested fields would have been shown, and how often it was right (first crop for a one-crop answer, one of the two for a two-crop answer).

| shown            | until 25 Nov             | 25 Nov, first crop right   | now, by situation        |
|:-----------------|:-------------------------|:---------------------------|:-------------------------|
| one crop, high   | 61% of fields, right 77% | 61% of fields, right 77%   | 61% of fields, right 77% |
| one crop, low    | 8% of fields, right 51%  | 4% of fields, right 55%    |                          |
| one crop, medium | 31% of fields, right 66% | 35% of fields, right 65%   | 31% of fields, right 67% |
| two crops, high  |                          |                            | 8% of fields, right 94%  |

## Replay on the map (season B, a point about every 1 km)

2,943 points. Before: the rule of the morning of 25 Nov (district hit rate, one step lower for a close call or data more than 5 km away). Now: this table.

| rule   | shown on the map   | Musanze   | Nyabihu   | Nyagatare   | Ruhango   |
|:-------|:-------------------|:----------|:----------|:------------|:----------|
| before | one crop, high     | 91%       |           | 78%         |           |
| before | one crop, low      |           | 15%       | under 1%    | 15%       |
| before | one crop, medium   | 9%        | 85%       | 22%         | 85%       |
| now    | no answer          |           |           | under 1%    |           |
| now    | one crop, high     | 91%       |           | 78%         |           |
| now    | one crop, medium   | 6%        | 95%       | 20%         | 90%       |
| now    | two crops, high    | 3%        | 5%        | 2%          | 10%       |

## How to read this

- These are tested rates, not promises: about 1 in 4 high-confidence answers was still wrong in testing.
- A two-crop answer means both crops fit the field about equally. The farmer can choose between them, for example by the seed and market they have.
- The rates within 5 km are conservative for a point inside a surveyed block: the model on the map has seen the block's fields, the tested models had not.
