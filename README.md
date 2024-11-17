# ncs-evidence-map

Code base to generate an evidence map of natural climate solutions related scientific papers and the evidence of co-impacts to people and nature. This repository accompanies [Chang et al. *Nature Sustainability* 2024](https://doi.org/10.1038/s41893-024-01454-z).

## Replication dataset

We share a replication dataset with the final sample of abstracts categorized to different NCS pathways and human and environmental co-impacts.

Below, we provide a description of the fields in the dataset, which can be viewed at the [`data`](https://github.com/lexunit-ai/ncs-evidence-map/tree/main/data) folder:

### Replication Dataset Fields

- **ESind**: Our index for this study.
- **abstract**: The article's abstract text.
- **addresses**: Author affiliations and institutional addresses.
- **articletitle**: Title of the article.
- **authors**: List of authors involved in the article.
- **biodiv_species_uid**: Unique identifier for biodiversity species mentioned in the article.
- **contains_cost_layer1**: Flag indicating whether cost information is included in the study.
- **cost_extraction_method**: Method used to extract cost data from the study.
- **doi**: Digital Object Identifier, a unique identifier for the article.
- **dup_title**: Duplicate title check for articles within the dataset.
- **efgs_filtered**: List of Ecosystem Functional Groups (EFGs) in the abstract.
- **functional_biomes_filtered**: List of functional biomes in an abstract.
- **esj_s1**: Ecosystem service category (primary).
- **esj_s2**: Ecosystem service category (secondary).
- **iplc_s1**: Indigenous Peoples and Local Communities (IPLC) category (primary).
- **iplc_s2**: Indigenous Peoples and Local Communities (IPLC) category (secondary).
- **issue**: Issue number of the publication.
- **language**: Language of the publication.
- **geolocation_status**: Status of geolocation data for the study.
- **location**: General geographic location of the study.
- **location_details**: Detailed location information of the study.
- **predicted_benefits**: Predicted co-impacts resulting from the ecosystem services.
- **predicted_pathway_numbers**: Numbers of predicted pathways for outcomes in the study.
- **publicationdate**: Date of publication.
- **publicationtype**: Type of publication (e.g., journal, conference).
- **publicationyear**: Year of publication.
- **pubmedid**: PubMed Identifier if applicable.
- **researchareas**: Thematic research areas covered by the article.
- **timescited**: Number of citations the article has received.
- **woscore**: Web of Science score for the article.
- **volume**: Volume number of the journal.
- **webofscienceindex**: Web of Science index or category in which the article is listed.

#### Dataset Integrity

To verify that the data has not been corrupted in the downloading process, users can check the validity of the downloaded file by navigating to the folder containing the data and running the following command:

`shasum -a 256 ncs-evidence-map-data.tsv`

This should produce the following checksum value: `cc899851a21d688a8e66c9748fcba6f2dbc8ab29040c2e177ead0b4e41ad5a15`

