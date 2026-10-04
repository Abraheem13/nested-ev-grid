# Reference verification ledger

Every entry of the submitted draft's bibliography (51 items) and every new
reference was checked against an independent record (publisher page, PMLR/ACM/
IEEE/Elsevier listing, arXiv, dblp, IDEAS/RePEc, institutional repository or
Google Research) via web search on 2026-10-03. The container had no direct
access to Crossref or doi.org, so DOIs are those reported by those records.

Legend: **OK** verified as cited; **FIXED** real work, citation details corrected;
**REMOVED** no record of the work could be found (likely fabricated) or the
citation did not support the sentence it was attached to.

## Items from the submitted draft

| Key | Status | Notes |
|---|---|---|
| iea2024 | FIXED | Draft used the *2024* outlook to support a *2025* sales figure. Replaced by IEA, *Global EV Outlook 2025* (sales > 17 million in 2024; > 20 million expected in 2025). |
| ieee1547 | FIXED (use) | Real standard, but it does not set the 0.95 p.u. service-voltage floor. The floor is ANSI C84.1-2020 Range A (±5 %); IEEE 1547-2018 is now cited only for inverter reactive-power capability (volt/var modes). |
| thurner2018 | OK | IEEE TPWRS 33(6):6510–6521, 2018, doi:10.1109/TPWRS.2018.2829021 (used only for validation of the power-flow solver). |
| baran1989 | OK | IEEE TPWRD 4(2):1401–1407, 1989, doi:10.1109/61.25627 (33-bus). |
| stiasny2021 | OK | Stiasny, Zufferey, Pareschi, Toffanin, Hug, Boulouchos (ETH Zürich), EPSR 191, 106696, 2021. |
| aljabri2026 | **REMOVED** | No record found for "Hierarchical deep reinforcement learning and model predictive control for voltage-aware EV charging coordination…", IEEE Access 14, 2026. |
| clement2010 | OK | IEEE TPWRS 25(1):371–380, 2010, doi:10.1109/TPWRS.2009.2036481. |
| ortegavazquez2014 | OK | IET GTD 8(6):1007–1016, 2014. |
| orfanoudakis2025 | FIXED | Venue is *Communications Engineering* (not "Nat. Commun. Eng."), vol. 4, Art. no. 118, 2025, doi:10.1038/s44172-025-00457-8; authors Orfanoudakis, Robu, Salazar, Palensky, Vergara. Article number added in the second check. |
| wang2023twostage (EV2Gym) | FIXED (key) | Orfanoudakis et al., IEEE T-ITS 26(2):2410–2421, 2025, doi:10.1109/TITS.2024.3510945. Key renamed `orfanoudakis2025ev2gym`. |
| mohsenian2010 | OK | IEEE TSG 1(3):320–331, 2010. |
| ma2013 | OK | IEEE TCST 21(1):67–78, 2013. |
| fescioglu2023 | FIXED (use) | Real (RSER 188, 113873, 2023, doi:10.1016/j.rser.2023.113873) but it is a review of ML methods, not "field evidence" that TOU cannot prevent voltage collapse; that sentence was removed. |
| wan2019 | FIXED (use) | Real (IEEE TSG 10(5):5246–5257, 2019, doi:10.1109/TSG.2018.2879572), but it is a DRL paper, not a droop controller as the draft stated. |
| zhang2020cddpg | OK | IEEE IoT J 8(5):3075–3087, 2021. |
| jin2020 | OK | IEEE TSG 12(2):1416–1428, 2021. |
| dasilva2020 | OK | IEEE TSG 11(3):2347–2356, 2020, doi:10.1109/TSG.2019.2952331. |
| liu2021madrl | **REMOVED** | No record found for "Multi-agent deep reinforcement learning for large-scale EV charging scheduling", IEEE TVT 70(11), 2021 (two searches). |
| qian2021 | OK | IEEE TSG 11(2):1714–1723, 2020. |
| park2022 | OK | Applied Energy 328, 120111, 2022. |
| zhang2023v2g | **REMOVED** | No record found for "Distributed training and distributed execution based Stackelberg multi-agent RL for EV charging scheduling", IEEE TSG 14(6), 2023. |
| shibl2023 | OK | Energy Reports 10:494–509, 2023. |
| tuchnitz2021 | OK | Applied Energy 285, 116382, 2021. |
| saner2022 | OK | IEEE TSG 13(3):2218–2233, 2022. |
| zhang2021p2p | **REMOVED** | No record found for "Distributed hierarchical coordination of networked charging stations based on peer-to-peer trading and EV charging flexibility quantification", IEEE TPWRS 37(4), 2022. |
| ye2022local | OK | Ye, Papadaskalopoulos, Yuan, Tang, Strbac, IEEE TSG 14(2), 2023. |
| pateria2021 | FIXED | ACM CSUR 54(5), Art. no. 109 (35 pp.), doi:10.1145/3453160. Year 2021 as in DBLP (journals/csur/PateriaSTQ21) and the article's own reference line (online 5 Jun. 2021); the printed issue is dated 2022. Article number added in the second check. |
| achiam2017 | OK | ICML 2017, PMLR 70:22–31. |
| su2025review | OK | Proc. IEEE 113(3):213–255, 2025. |
| wang2020voltvar | OK | IEEE TSG 11(4):3008–3018, 2020. |
| kou2020 | OK | Applied Energy 264, 114772, 2020, doi:10.1016/j.apenergy.2020.114772. |
| gao2022modelaug | OK | Applied Energy 313, 118762, 2022. |
| shi2022stability | FIXED | Y. Shi, G. Qu, S. Low, A. Anandkumar, A. Wierman, ACC 2022, pp. 2715–2721, doi:10.23919/ACC53348.2022.9867476. Pages added in the second check. |
| wang2024safemarl | FIXED | Authors are Y. Qu, J. Ma, F. Wu (not "Y. Wang et al."); IJCAI 2024, pp. 184–192, doi:10.24963/ijcai.2024/21. |
| yu2024safereview | FIXED | Now cites the peer-reviewed version: P. Yu, H. Zhang, Y. Song, Z. Wang, H. Dong, L. Ji, *Renew. Sustain. Energy Rev.* 223, Art. no. 116022, 2025, doi:10.1016/j.rser.2025.116022 (RePEc rensus/v223y2025ics1364032125006951; preprint arXiv:2407.00681 had four authors). |
| hu2022voltage | OK | D. Hu, Z. Ye, Y. Gao, Z. Ye, Y. Peng, N. Yu, IEEE TSG 13(6):4873–4886, 2022. |
| cao2024pignn | OK | IEEE TSG 15(1):233–246, 2024. |
| gao2021consensus | OK | IEEE TSG 12(4):3594–3604, 2021. |
| raissi2019pinn | OK | J. Comput. Phys. 378:686–707, 2019. |
| wang2023coord | OK | IEEE TII 19(2):1611–1622, 2023. |
| hu2021dmpc | OK | Hu et al., IEEE TSG 13(1):576–588, 2022 (cited by later work). |
| mazumder2021 | **REMOVED** | No record found for "EV charging stations with a provision of V2G and voltage support in a distribution network", IEEE Syst. J. 15(1), 2021. |
| mattos2024 | OK | IET GTD 18(6):1133–1157, 2024, doi:10.1049/gtd2.13066. |
| leemput2015 | FIXED | Authors are N. Leemput, F. Geth, J. Van Roy, J. Büscher, J. Driesen (draft listed "P. Vanhoutte"); SEGN 3:24–35, 2015. |
| singh2013 | OK | IEEE TSG 4(2):1026–1037, 2013, doi:10.1109/TSG.2013.2238562. |
| behrouz2025 | OK | NeurIPS 2025 (arXiv:2512.24695). |
| schulman2017 | OK | arXiv:1707.06347. |
| lillicrap2016 | OK | ICLR 2016 (arXiv:1509.02971). |
| fishbein1975 | OK | Addison-Wesley, 1975. |
| lee2019acn | OK | e-Energy '19, pp. 139–149, doi:10.1145/3307772.3328313. |
| nrel2015 | **REMOVED** | The NREL EV Project data is no longer used (no session data was ever loaded); the report number could not be confirmed. |

## New references (data sources, methods)

| Key | Status | Notes |
|---|---|---|
| iea2025 | OK | IEA, *Global EV Outlook 2025*, Paris, 2025. |
| ansi_c84 | OK | ANSI C84.1-2020, *Electric Power Systems Voltage Ratings (60 Hz)*, NEMA, 2020 (Range A ±5 %). |
| baran1989cap | OK | IEEE TPWRD 4(1):725–734, 1989, doi:10.1109/61.19265 (69-bus). |
| zimmerman2011 | OK | MATPOWER, IEEE TPWRS 26(1):12–19, 2011, doi:10.1109/TPWRS.2010.2051168. |
| shirmohammadi1988 | OK | IEEE TPWRS 3(2):753–762, 1988 (backward/forward sweep). |
| hirth2018 | OK | ENTSO-E Transparency Platform review, Applied Energy 225:1054–1067, 2018, doi:10.1016/j.apenergy.2018.04.048. |
| pecanstreet | OK | Pecan Street Inc., Dataport (attribution required). |
| yeh2023sustaingym | OK | SustainGym, NeurIPS 2023 Datasets and Benchmarks (Yeh et al.). |
| xu2016priority | OK | Y. Xu, F. Pan, L. Tong, "Dynamic scheduling for charging electric vehicles: A priority rule," IEEE TAC 61(12), pp. 4094–4099, Dec. 2016 (least-laxity-first principle; also arXiv:1602.00372; IEEE Xplore document 7431973). Pages added in the second check. |
| huangfu2018 | OK | HiGHS, Math. Prog. Comp. 10(1):119–142, 2018, doi:10.1007/s12532-017-0130-5. |
| gupta2017 | OK | Gupta, Egorov, Kochenderfer, ALA workshop at AAMAS 2017 (parameter sharing). |
| silver2018residual | OK | T. Silver, K. Allen, J. Tenenbaum, L. Kaelbling, "Residual policy learning," arXiv:1812.06298, 2018. |
| johannink2019residual | OK | T. Johannink et al., "Residual reinforcement learning for robot control," ICRA 2019, pp. 6023–6029. |
| stooke2020pid | OK | A. Stooke, J. Achiam, P. Abbeel, "Responsive safety in reinforcement learning by PID Lagrangian methods," ICML 2020, PMLR 119:9133–9143. |
| orfanoudakis2025ev2gym | OK | EV2Gym (previously cited under the key wang2023twostage): Orfanoudakis, Diaz-Londono, Yılmaz, Palensky, Vergara, IEEE T-ITS 26(2):2410–2421, 2025. |

## Notes on the final bibliography (`paper/references.tex`)

* Removed entries (no record found or not supporting the claim): aljabri2026,
  liu2021madrl, zhang2023v2g, zhang2021p2p, mazumder2021, nrel2015; iea2024 was
  replaced by iea2025.
* Full author lists were written out and checked for: johannink2019residual
  (Johannink, Bahl, Nair, Luo, Kumar, Loskyll, Ojea, Solowjow, Levine),
  cao2024pignn (D. Cao, J. Zhao, J. Hu, Y. Pei, Q. Huang, Z. Chen, W. Hu),
  dasilva2020 (Da Silva, Nishida, Roijers, Costa), yeh2023sustaingym (12
  authors, NeurIPS 2023 Datasets and Benchmarks), gupta2017 (LNCS 10642,
  pp. 66–83).
* Second, independent check (all 58 entries re-searched; direct database APIs
  were blocked by the network policy, so records were confirmed through search
  results quoting IEEE Xplore, ScienceDirect, ACM DL, Springer, PMLR, IJCAI,
  NeurIPS and author pages): no entry was found wrong; the page and article
  numbers of xu2016priority, shi2022stability, orfanoudakis2025 and
  pateria2021 were added, and yu2024safereview now cites its journal version.
* `pecanstreet` is an online resource without an access date: the data were
  obtained through the EV2Gym distribution, not downloaded from Dataport.
* `scripts/check_paper.py` fails if a cited key is not marked OK or FIXED above.
