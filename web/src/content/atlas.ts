/**
 * Field guide: the eight classes in the two training datasets, in plain words. Descriptions are general plant
 * pathology (what the condition is and what it looks like), not results of this project. `key` matches
 * site.json `dataset.by_class`; `classes` lists the `orig_class` spellings used by the sample photos.
 */
export type Tissue = 'yellow' | 'brown' | 'dark' | 'pale' | 'none';
export type AtlasEntry = {
  key: string;
  crop: 'turmeric' | 'blackgram';
  classes: string[];
  name: string;
  healthy: boolean;
  cause: string;
  looks: string;
  /** which of the pipeline's abnormal-tissue colours this condition mostly shows up as */
  tissue: Tissue[];
  reads: string;
};

export const TISSUE_LABEL: Record<Tissue, string> = {
  yellow: 'yellowing',
  brown: 'brown dead tissue',
  dark: 'dark tissue',
  pale: 'pale, mildew-like',
  none: 'no abnormal colour',
};
export const TISSUE_COLOUR: Record<Tissue, string> = {
  yellow: 'var(--chlorosis)',
  brown: 'var(--necrosis)',
  dark: 'var(--dark-tissue)',
  pale: 'var(--mildew)',
  none: 'var(--leaf)',
};

export const ATLAS: AtlasEntry[] = [
  {
    key: 'turmeric/healthy_leaf',
    crop: 'turmeric',
    classes: ['healthy_leaf', 'healthy'],
    name: 'Healthy',
    healthy: true,
    cause: 'No disease',
    looks: 'A long, even green blade with a strong midrib and fine parallel veins. Glare and vein shadows are the only things that break the colour.',
    tissue: ['none'],
    reads: 'The baseline. The pipeline learns this leaf’s own green first, then looks for tissue that departs from it.',
  },
  {
    key: 'turmeric/leaf_blotch',
    crop: 'turmeric',
    classes: ['leaf_blotch', 'blotch'],
    name: 'Leaf blotch',
    healthy: false,
    cause: 'Fungus · Taphrina maculans',
    looks: 'Many small, pale yellow spots on both sides of the leaf. They darken to dirty yellow and brown, run together into irregular blotches, and the leaf ends up looking scorched.',
    tissue: ['yellow', 'brown'],
    reads: 'Lots of small regions scattered across the blade: a high spot count with yellow and brown tissue together.',
  },
  {
    key: 'turmeric/dry_leaf',
    crop: 'turmeric',
    classes: ['dry_leaf'],
    name: 'Dry leaf',
    healthy: false,
    cause: 'Drying and scorch · often the last stage of leaf spot or blotch, also age or water stress',
    looks: 'Brown, papery tissue that starts at the tip and edges and moves inward. Large areas lose their green entirely.',
    tissue: ['brown', 'dark'],
    reads: 'A few very large regions rather than many small ones, so the damaged share of the leaf is high.',
  },
  {
    key: 'blackgram/healthy',
    crop: 'blackgram',
    classes: ['healthy'],
    name: 'Healthy',
    healthy: true,
    cause: 'No disease',
    looks: 'Three broad, oval leaflets with a net of veins and a slightly hairy, matt surface. Colour is an even mid green.',
    tissue: ['none'],
    reads: 'Field photos add soil, stems and neighbouring plants, so finding the leaf is the first job here.',
  },
  {
    key: 'blackgram/yellow_mosaic',
    crop: 'blackgram',
    classes: ['yellow_mosaic'],
    name: 'Yellow mosaic',
    healthy: false,
    cause: 'Virus · mungbean yellow mosaic virus, carried by whiteflies',
    looks: 'Irregular bright yellow patches between green ones, like a mosaic. The yellow spreads until whole leaflets turn golden.',
    tissue: ['yellow'],
    reads: 'The clearest colour signal of all: large areas shift from green toward yellow while the leaf keeps its shape.',
  },
  {
    key: 'blackgram/anthracnose',
    crop: 'blackgram',
    classes: ['anthracnose'],
    name: 'Anthracnose',
    healthy: false,
    cause: 'Fungus · Colletotrichum species',
    looks: 'Round to angular brown spots with darker edges, often sunken. Centres can dry and drop out, leaving holes in the leaflet.',
    tissue: ['brown', 'dark'],
    reads: 'Compact, fairly round regions with sharp borders: spot shape and dark tissue carry the reading.',
  },
  {
    key: 'blackgram/powdery_mildew',
    crop: 'blackgram',
    classes: ['powdery_mildew'],
    name: 'Powdery mildew',
    healthy: false,
    cause: 'Fungus · Erysiphe polygoni',
    looks: 'White, flour-like patches on the upper surface that grow until they coat the leaflet, which later dulls and browns.',
    tissue: ['pale'],
    reads: 'Tissue turns paler and less saturated instead of yellow or brown, which is why pale tissue has its own colour class.',
  },
  {
    key: 'blackgram/leaf_crinckle',
    crop: 'blackgram',
    classes: ['leaf_crinckle'],
    name: 'Leaf crinkle',
    healthy: false,
    cause: 'Virus · urdbean leaf crinkle virus, seed-borne',
    looks: 'Leaflets grow oversized, puckered and curled, with a rough, wrinkled surface. The colour can stay close to healthy green.',
    tissue: ['none'],
    reads: 'A change of surface more than of colour, so texture measurements matter here and the colour rules have little to hold on to.',
  },
];
