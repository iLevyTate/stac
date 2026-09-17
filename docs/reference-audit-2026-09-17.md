# Reference audit across the three published chapters and the exegesis, 2026-09-17

Every DOI and arXiv identifier in the reference lists of the three published chapters was resolved
(Crossref for the chapter lists, doi.org and the arXiv export API for the identifiers, DataCite for
Zenodo), and every claimed title was searched in Crossref. The exegesis list was checked the same way
from the June 2026 PDF. The check the corrigendum had done before this covered only the Zenodo
entries. That was not enough.

Classification used below: **fabricated** means no work with that title exists in Crossref or arXiv
and the identifier points at an unrelated paper; **real, wrong identifier** means the work exists and
the entry carries the wrong DOI, venue, year or author list; **dead** means the DOI does not resolve
and the work's existence could not be confirmed either way.

## Aligned Minds, Efficient Machines (IGI ch005, DOI 10.4018/979-8-3373-5702-7.ch005)

Beyond the two entries already in the correction request (the truncated `140532` and the duplicated
`15867066`), seven more of the 40 references are defective. Each is cited once in the body.

| # | Entry as published (p. 140–143) | Identifier resolves to | Classification | Cited to support |
|---|---|---|---|---|
| 1 | Kim, C., Johnson, N. F., & Gold, B. T. (2011). Cingulo-frontal network activates differentially to conscious and non-conscious conflict. *PNAS, 108*(42), 17547–17552. DOI 10.1073/pnas.1103627108 | Christiansen et al., RAS-converting enzyme 1, PNAS 2011 | **Fabricated.** No such title anywhere. The real Kim, Johnson, Cilles & Gold (2011) is "Common and distinct mechanisms of cognitive flexibility in prefrontal cortex", *J. Neurosci. 31*(13), 4771–4779, DOI 10.1523/JNEUROSCI.5923-10.2011, which the exegesis cites correctly. | ACC error-monitoring, p. 125 |
| 2 | Li, S., Wang, X., Sun, Y., Wu, P., & Chen, A. (2021). The ventromedial prefrontal cortex in emotion regulation: Combined TMS-fMRI evidence. *Cerebral Cortex, 31*(12), 5482–5496. DOI 10.1093/cercor/bhab139 | Mazuir et al., oligodendrocyte secreted factors, Cereb. Cortex 2021 | **Fabricated.** No such title; no such author set. | VMPFC "gut feeling", p. 123 |
| 3 | Ling, Y., Zhang, J., & Wang, K. (2022). Computer-adaptive forms for multidimensional questionnaires: Methodological advances and simulation results. *Applied Psychological Measurement, 46*(5), 401–417. DOI 10.1177/01466216211073012 | Nothing (404) | **Fabricated.** No such title in Crossref. | "As validated extensively in psychometric literature (Ling et al., 2022)", p. 129 |
| 4 | Rolls, E. T. (2023). The orbitofrontal cortex, amygdala, reward value, emotion, and decision-making. *Neuroscience and Biobehavioral Reviews, 150*, 105066. DOI 10.1016/j.neubiorev.2023.105066 | Fricke & Vogel, a corrigendum, NBR 2023 | **Real, wrong identifier.** That title is a 2023 OUP book chapter (10.1093/oso/9780198887911.003.0011). The journal paper is Rolls (2023), "Emotion, motivation, decision-making, the orbitofrontal cortex, anterior cingulate cortex, and the amygdala", *Brain Struct. Funct.*, 10.1007/s00429-023-02644-9, which the exegesis cites correctly. | OFC value updating, p. 124 |
| 5 | Sullivan, J. G., Vu, M. A. T., Stein, D. J., & Yazdanfar, A. (2022). Medial prefrontal cortex activation during third-party social interactions in infants. *Developmental Science, 25*(5), e13245. DOI 10.1111/desc.13245 | Sullivan, Xie et al., inhibitory control in Bangladeshi children, Dev. Sci. 2022 | **Fabricated author list on a real finding.** The finding is Farris, Kelsey, Krol, Thiele, Hepach, Haun & Grossmann (2022), "Processing third-party social interactions in the human infant brain", *Infant Behav. Dev. 68*, 101727, 10.1016/j.infbeh.2022.101727, which the exegesis cites correctly. | mPFC perspective-taking, p. 126 |
| 6 | Zenke, F., & Vogels, T. P. (2021). … *Nature Machine Intelligence, 3*(1), 76–87. DOI 10.1038/s42256-020-00286-3 | Nothing (404) | **Real, wrong identifier.** *Neural Computation, 33*(4), 899–925, DOI 10.1162/neco_a_01367. Exegesis has it right. | Surrogate gradients, p. 133 |
| 7 | Fang, W., Yu, Z., Chen, Y., Masquelier, T., Huang, T., & Tian, Y. (2021). SpikingJelly: A deep-learning framework for spiking neural networks. *Frontiers in Neuroscience, 15*, 774067. DOI 10.3389/fnins.2021.774067 | Nothing (404) | **Real, wrong identifier.** Fang, Chen, Ding et al. (2023), "SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence", *Science Advances, 9*(40), eadi1480, DOI 10.1126/sciadv.adi1480. | STAC V2 tooling, p. 135 |

The remaining 31 entries resolve to the works they name. The Vaswani et al. (2017) entry is garbled by
typesetting but retrievable.

## Beyond Intelligence (IGI ch007, DOI 10.4018/979-8-3373-4252-8.ch007)

Confirmed on the published PDF (printed pp. 257–286, supplied 2026-09-17); all eight entries appear
as below on printed pp. 281–284. Eight defective.

| # | Entry | Identifier resolves to | Classification | Cited |
|---|---|---|---|---|
| 1 | Farrell, T., & Yu, P. L. H. (2020). Personality-based adaptive human-AI interaction: An overview. *IJHCS, 139*, 102428. 10.1016/j.ijhcs.2020.102428 | Vi et al., LeviSense (levitating food), IJHCS 2020 | **Fabricated.** | §6.2, once |
| 2 | Grossberg, S. (2021). Conscious MIND resonates with attentive ART: Toward biologically plausible machine learning. *Neural Computation, 33*(10), 2583–2678. 10.1162/neco_a_01417 | Mofrad et al., associative memory, Neural Comput. 2021 | **Fabricated article form of a real book.** Grossberg (2021), *Conscious Mind, Resonant Brain*, OUP, 10.1093/oso/9780190070557.001.0001; the exegesis cites the book. | §2.1, once |
| 3 | Gudmundsson, E., & Lönner, V. J. (2009). Cross-cultural adaptation of psychological scales. In Gerstein et al. (Eds.), *International handbook of cross-cultural counseling* (pp. 123–141). SAGE. 10.4135/9781483328914.n8 | Nothing (404) | **Dead.** The handbook exists; this chapter and page range could not be confirmed. Not cited in the body. | 0× |
| 4 | Leng, J., Zhang, H., Yan, D., Liu, Q., Chen, X., & Zhang, D. (2022). Digital twin-driven manufacturing cyber-physical system: A survey. *J. Manuf. Syst., 62*, 493–512. 10.1016/j.jmsy.2021.12.012 | Bai et al., LMPF cloud manufacturing, JMS 2022 | **Fabricated title on real authors.** Leng et al. (2021), "Digital twins-based smart manufacturing system design in Industry 4.0: A review", *JMS 60*, 119–137, 10.1016/j.jmsy.2021.05.011, is real and is what the exegesis cites. | §6, 4× |
| 5 | Ling, Y., Guo, X., Luo, X., & Liu, C. (2022). Development of a computerized adaptive test for problematic mobile phone use. *Front. Psychol., 13*, 837618. 10.3389/fpsyg.2022.837618 | Nothing (404) | **Real, wrong identifier.** Liu, Lu, Zhou et al. (2022), same title, *Front. Psychol. 13*, 892387, 10.3389/fpsyg.2022.892387. Author list differs. Not cited in the body. | 0× |
| 6 | Shoeybi, M., et al. (2019). Megatron-LM … *Proc. SC'19*. 10.1145/3295500.3356181 | Kwasniewski et al., "Red-blue pebbling revisited", SC'19 | **Real, wrong identifier.** Megatron-LM is arXiv:1909.08053 and was never in the SC'19 proceedings. | §5, 2× |
| 7 | Roumeliotis, T., & Tselikas, N. D. (2023). Effective CLI design for modular AI systems. arXiv. 10.48550/arXiv.2109.09331 | Meyer-Vitali & Mulder, "Modular Design Patterns for Hybrid Actors" | **Fabricated.** Real Roumeliotis & Tselikas (2023) is "ChatGPT and Open-AI models: A preliminary review", *Future Internet 15*(6), 192, which the Springer chapter and the exegesis cite correctly. | §5.1, once |
| 8 | Schmidgall, S., Smith, J., & Patel, R. (2024). Brain-inspired learning and plasticity in artificial neural networks. arXiv. 10.48550/arXiv.2403.12345 | Tramm et al., Monte Carlo particle transport on GPUs | **Fabricated coauthors and identifier on a real review.** Schmidgall, Ziaei, Achterberg, Kirsch, Hajiseyedrazi & Eshraghian (2024), "Brain-inspired learning in artificial neural networks: A review", *APL Machine Learning 2*(2), 021501, 10.1063/5.0186054 (arXiv:2305.11252). | §2.1, once |

The other 54 entries resolve correctly, including all four Zenodo DOIs.

## Synthetic Cognitive Augmentation Network (Springer SEET 2025, DOI 10.1007/978-3-032-08977-9_13)

One of 22 entries. Ref. 9: "Zhang, S., Dean, J.C.: Spikeformers: Transformers with Spiking Neural
Networks. arXiv preprint arXiv:2109.12894 (2021)". That identifier is Eshraghian et al., "Training
Spiking Neural Networks Using Lessons From Deep Learning". No work titled "Spikeformers" by those
authors exists. **Fabricated.** The other 21 resolve.

## The exegesis (June 2026 PDF)

Its 27 DOIs and 4 arXiv identifiers were resolved. It gets right several entries the chapters get
wrong (Kim et al. 2011, Rolls 2023, Farris et al. 2022, Zenke & Vogels, Grossberg's book, Leng et al.
2021, Roumeliotis & Tselikas). It repeats three defects: the Li/Chen (2021) TMS-fMRI entry (without a
DOI), the SpikingJelly *Frontiers* entry, and the Schmidgall/Smith/Patel arXiv entry. It also carries
two stale placeholders in a June 2026 document, "Kennedy, B., Mohammad, A., & Wyandt, M. (2023a).
Synthetic cognitive augmentation network [Pending final release information]" and "Kennedy, B. (2025a).
Aligned minds and efficient machines … [Manuscript in preparation]", the second of which duplicates
the correct "Kennedy, B. (2026)" entry two lines below it. And it reproduces Appendix A in full, so
the BRIEF-A exposure recorded in SCAN-Resources `PROVENANCE.md` applies to it as well.

## What this means

The pattern (plausible authors, plausible titles, identifiers that point at unrelated papers in the
same journal or the same arXiv month) is the signature of citations produced without retrieval and
never checked. Sixteen of them across three published chapters. The chapters' arguments do not rest
on any of them; every one supports a background sentence that a real citation could support instead,
and in eleven of the sixteen a real replacement is named above.

Each publisher needs a correction. For IGI ch005 the seven entries can go into the marked copy as
group B, which becomes nine comments instead of two; the fix in every case is a replacement reference
entry, and the digital copy is what the publisher has said it can change. For IGI ch007 and for
Springer this is the first correction request, and it should be sent as its own letter.

## Marked copy for ch007, built 2026-09-17

`Beyond-Intelligence-CORRECTIONS-MARKED.pdf`: a summary note on printed p. 257 and eight highlight
comments on pp. 281–284, one per entry. Five ask for a replacement entry (Grossberg's book;
Leng et al. 2021, 10.1016/j.jmsy.2021.05.011; Liu et al. 2022, 10.3389/fpsyg.2022.892387; Megatron-LM
as arXiv:1909.08053; Schmidgall et al. 2024, 10.1063/5.0186054), two ask for deletion with a body
change (Farrell & Yu → Siemon et al. 2022, 10.3390/su14073832, on p. 274; Roumeliotis & Tselikas
citation removed from the sentence on p. 266), one asks for deletion with no other change
(Gudmundsson & Lönner, uncited). "(Leng et al., 2022)" changes year on pp. 259, 272, 275 and 276.
Every replacement DOI was resolved and its author list taken from Crossref before the file was built.
Not yet sent.
