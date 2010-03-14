package mulan.classifier.meta;

import mulan.classifier.transformation.LabelPowerset;
import weka.classifiers.trees.J48;
import weka.clusterers.SimpleKMeans;
import weka.core.ManhattanDistance;

public class ClusteringBasedTest extends MultiLabelMetaLearnerTest {

    @Override
    public void setUp() throws Exception {
        SimpleKMeans clusterer = new SimpleKMeans();
        clusterer.setNumClusters(5);
        clusterer.setDistanceFunction(new ManhattanDistance());
        learner = new ClusteringBased(clusterer, new LabelPowerset(new J48()));
    }
}
