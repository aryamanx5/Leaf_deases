/** Plain-language explanations behind the "?" buttons on /analyze. */
export const GLOSSARY = {
  cnn: {
    term: 'Neural network (CNN)',
    text: 'A convolutional neural network: a program that learned to spot disease by studying hundreds of labelled leaf photos. It looks at the pixels of the photo directly.',
  },
  svm: {
    term: 'Machine-learning reading (SVM)',
    text: 'A support vector machine, a classic machine-learning model. It never sees the photo: it reads the measurements image processing took (colour, damaged area, texture) and learned where healthy ends and diseased begins.',
  },
  rule: {
    term: 'Rule-based score',
    text: 'A hand-written formula: it adds up four measurements (damaged area, number of spots, abnormal colour, texture) and compares the total with a threshold chosen on validation photos.',
  },
  segmentation: {
    term: 'Leaf cut-out (segmentation)',
    text: 'Separating the leaf from its background, so only the leaf is measured and the table or soil behind it is ignored.',
  },
  lesion: {
    term: 'Damaged spots (lesions)',
    text: 'Patches whose colour differs from this leaf’s own healthy green: yellowing, brown dead tissue, dark tissue or pale, mildew-like patches.',
  },
  gradcam: {
    term: 'Where the AI looked (Grad-CAM)',
    text: 'Gradient-weighted class activation map. It colours the parts of the photo that pushed the neural network, the AI, towards its answer: red most, blue least. It shows attention, not damage.',
  },
  ood: {
    term: 'Unusual photo check (OOD)',
    text: 'Out-of-distribution check: how different this photo is from the photos the model learned from. A very unusual photo (another plant, a drawing, poor light) makes every answer less reliable.',
  },
  glcm: {
    term: 'Texture (GLCM)',
    text: 'Grey-level co-occurrence matrix: a standard way to measure texture, that is how much neighbouring pixels differ. Damaged leaves are usually rougher.',
  },
  methods: {
    term: 'Two methods',
    text: 'Patra reads every leaf in two independent ways: image processing measures the leaf and a machine-learning model (SVM) judges those measurements; a neural network (CNN) judges the photo itself. When both agree, the answer is more trustworthy.',
  },
  confidence: {
    term: 'How sure is this?',
    text: 'The level combines how sure the neural network is (its calibrated probability) with whether the two methods agree and whether the photo looks normal.',
  },
} as const;

export type GlossaryKey = keyof typeof GLOSSARY;
