/**
 * Cited background facts (Section 6.3). These four statistics come from the project's
 * fellowship proposal; each was traced to the source below. No other statistics are used.
 */
export interface Citation {
  id: string;
  text: string;
  value: string;
  source_title: string;
  source_url: string | null;
  note: string;
}

export const citations: Citation[] = [
  {
    id: 'turmeric-blotch-loss',
    text: 'yield loss in turmeric from leaf blotch',
    value: '37.6–52.9%',
    source_title:
      'Chopada & Rakholiya (2015), Assessment of yield losses due to leaf blotch of turmeric caused by Taphrina maculans Butler, The Bioscan',
    source_url: 'https://thebioscan.com/index.php/pub/article/view/1893',
    note: 'Range reported there from Panja et al. (2000). Applies to leaf blotch (Taphrina maculans).',
  },
  {
    id: 'blackgram-ymd-loss',
    text: 'crop loss in black gram from yellow mosaic disease',
    value: '5–100%',
    source_title:
      'Vadivel, Manivannan et al. (2019), Correlation analysis for yield, yield components and MYMV disease scores in blackgram, Electronic Journal of Plant Breeding 10(2): 712–716',
    source_url: 'https://doi.org/10.5958/0975-928X.2019.00091.7',
    note: 'Range attributed there to Nene (1972); depends on severity, cultivar and whitefly pressure.',
  },
  {
    id: 'india-turmeric-share',
    text: "of the world's turmeric is grown in India",
    value: '>75%',
    source_title: 'Spices Board India, Indian spices brochure',
    source_url: 'https://indianspices.com/sites/default/files/brochure%20NZD.pdf',
    note: 'The brochure gives around 80% of world production; the proposal states it conservatively as >75%.',
  },
  {
    id: 'blackgram-pulse-area',
    text: "of India's pulse area is black gram",
    value: '~29%',
    source_title: 'ANGRAU Agricultural Market Intelligence Centre, Blackgram Outlook Report (January–December 2022)',
    source_url:
      'https://angrau.ac.in/downloads/AMIC/OutlookReports/2022/7-Blackgram%20Outlook%20Report%20-%20January%20to%20December%202022.pdf',
    note: 'States that black gram area accounts for about 29 per cent of India’s total pulse acreage.',
  },
];

export const datasetCredits = [
  {
    name: 'Black gram Plant Leaf Disease (BPLD)',
    by: 'Talasila et al., Data in Brief 45 (2022) 108725',
    url: 'https://data.mendeley.com/datasets/zfcv9fmrgv/3',
  },
  {
    name: 'Turmeric Plant Disease Dataset',
    by: 'Data in Brief 59 (2025) 111435',
    url: 'https://data.mendeley.com/datasets/g46dvrcvwn/2',
  },
  {
    name: 'Field-photo evaluation set',
    by: 'Mendeley (CC BY 4.0), iNaturalist and Wikimedia Commons contributors (CC0 / CC BY / CC BY-SA)',
    url: null,
    // where the field photos came from (configs/default.yaml "wild"; per-photo sources in data/wild/SOURCES.csv)
    links: [
      { label: 'Turmeric field photos (Mendeley)', url: 'https://doi.org/10.17632/jtttfbx342.2' },
      { label: 'Black gram field photos (Mendeley)', url: 'https://doi.org/10.17632/z55yrbmn2d.3' },
      { label: 'iNaturalist: turmeric', url: 'https://www.inaturalist.org/observations?taxon_name=Curcuma%20longa' },
      { label: 'iNaturalist: black gram', url: 'https://www.inaturalist.org/observations?taxon_name=Vigna%20mungo' },
      { label: 'Wikimedia Commons: turmeric', url: 'https://commons.wikimedia.org/wiki/Category:Curcuma_longa' },
      { label: 'Wikimedia Commons: black gram', url: 'https://commons.wikimedia.org/wiki/Category:Vigna_mungo' },
    ],
  },
] as { name: string; by: string; url: string | null; links?: { label: string; url: string }[] }[];
