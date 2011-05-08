package mulan.classifier.meta;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.ConditionalDependenceIdentifier;
import mulan.data.GreedyLabelClustering;
import weka.classifiers.trees.J48;

public class SubsetLearnerWithGreedyClustering_ConditionalTest extends MultiLabelMetaLearnerTest {

	@Override
	public void setUp() throws Exception {
		MultiLabelLearner lp = new LabelPowerset(new J48());
		ConditionalDependenceIdentifier cond = new ConditionalDependenceIdentifier(new J48());
		cond.setNumFolds(2);
		GreedyLabelClustering greedy = new GreedyLabelClustering(lp, new J48(), cond);
		greedy.setNumFolds(2);
		learner = new SubsetLearner(greedy, lp, new J48());
		learner.setDebug(true);
	}
}