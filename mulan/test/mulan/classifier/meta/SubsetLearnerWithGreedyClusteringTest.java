package mulan.classifier.meta;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.GreedyLabelClustering;
import mulan.data.LabelPairsDependenceIdentifier;
import mulan.data.UnconditionalChiSquareIdentifier;
import weka.classifiers.trees.J48;

public class SubsetLearnerWithGreedyClusteringTest extends MultiLabelMetaLearnerTest {

	@Override
	public void setUp() throws Exception {
		MultiLabelLearner lp = new LabelPowerset(new J48());
		LabelPairsDependenceIdentifier uncond = new UnconditionalChiSquareIdentifier();
		GreedyLabelClustering greedy = new GreedyLabelClustering(lp, new J48(), uncond);
		greedy.setNumFolds(2);
		learner = new SubsetLearner(greedy, lp, new J48());
		learner.setDebug(true);
	}

}