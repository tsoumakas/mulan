package mulan.classifier.meta;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.ConditionalDependenceIdentifier;
import weka.classifiers.trees.J48;

public class EnsembleOfSubsetLearners_ConditionalTest extends MultiLabelMetaLearnerTest {

	@Override
	public void setUp() throws Exception {
		ConditionalDependenceIdentifier cond = new ConditionalDependenceIdentifier(new J48());
		cond.setNumFolds(2);
		MultiLabelLearner lp = new LabelPowerset(new J48());
		learner = new EnsembleOfSubsetLearners(lp, new J48(), cond, 10);
		((EnsembleOfSubsetLearners)learner).setNumModels(2);
	}

}