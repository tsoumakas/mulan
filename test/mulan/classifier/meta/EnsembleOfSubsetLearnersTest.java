package mulan.classifier.meta;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.LabelPairsDependenceIdentifier;
import mulan.data.UnconditionalChiSquareIdentifier;
import weka.classifiers.trees.J48;

public class EnsembleOfSubsetLearnersTest extends MultiLabelMetaLearnerTest {

	@Override
	public void setUp() throws Exception {
		MultiLabelLearner lp = new LabelPowerset(new J48());
		LabelPairsDependenceIdentifier uncond = new UnconditionalChiSquareIdentifier();
		learner = new EnsembleOfSubsetLearners(lp, new J48(), uncond, 10);
		((EnsembleOfSubsetLearners)learner).setNumModels(2);
	}

}