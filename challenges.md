# Challenges

One new chat per line (`uv run neova chat`). Type the customer's name to identify.

| # | Challenge | Client | Message(s) | Expected |
|---|---|---|---|---|
| 1 | Answer from the documents | — | `Combien coûte la fibre 1 Gb/s ?` | 39,99 € / mois |
| 1 | Scanned image | — | `Combien coûte l'utilisation de mon mobile en Europe ?` | Answer from the roaming sheet |
| 2 | Contradiction | — | `La fibre 1 Gb/s est-elle toujours en promo ?` | No, current price 39,99 € |
| 2 | Contradiction | — | `Quels étaient les prix de la promo de rentrée 2024 ?` | 2024 prices, presented as a past offer |
| 3 | No answer | — | `Quels sont les horaires de votre boutique à Lyon ?` | Transfer, or "je n'ai pas cette information"; never invented |
| 4 | API as a tool | Patrick Doré | `Patrick Doré` then `Combien je dois ?` | 39,99 € |
| 5 | Validation | — | `Je veux un technicien.` | Asks for identification, books nothing |
| 5 | Confirmation | Patrick Doré | `Patrick Doré`, `Je n'ai plus internet`, then `oui` | Slot proposed with fees, then "C'est confirmé…" only after `oui` |
| 5 | API failure | Patrick Doré | As above, API failures on before `oui`, off, then `alors ?` | "Je n'ai pas pu confirmer…", then confirmed once |
| 6 | Immediate transfer | — | `Je vais saisir mon avocat.` | Transfer without a single question |
| 6 | Transfer after review | Ahmed Belkacem | `Ahmed Belkacem` then `Je peux payer en plusieurs fois ?` | Transfer (payment plan) |
| 7 | API unavailable | Camille Rousseau | API failures on, then `Camille Rousseau` | "Service indisponible", no crash, no invented data |
| 8 | Internet failure | Camille Rousseau | `Camille Rousseau` then `Internet ne marche plus` | Mentions the ongoing outage in her area |
| 8 | Billing | Ahmed Belkacem | `Ahmed Belkacem` then `Je conteste ma facture, on m'a prélevé deux fois` | Transfer (dispute) |
| 8 | Moving house | — | `Je déménage, est-ce que mon engagement repart à zéro ?` | No, the commitment continues |
| 8 | Termination | Ahmed Belkacem | `Ahmed Belkacem` then `Combien je paie si je résilie maintenant ?` | Fees computed with the CGV rule |

API failures on / off, in another terminal while the chat is running:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/admin/chaos -ContentType application/json -Body '{"rate": 1}'
Invoke-RestMethod -Method Post http://localhost:8000/admin/chaos -ContentType application/json -Body '{"rate": 0}'
```

Rate limits (429 / 529) cannot be tested by hand; they are covered by `uv run pytest -q`.
