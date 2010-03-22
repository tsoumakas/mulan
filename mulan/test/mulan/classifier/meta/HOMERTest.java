package mulan.classifier.meta;

import mulan.classifier.transformation.LabelPowerset;
import org.junit.Ignore;
import weka.classifiers.trees.J48;

@Ignore
public class HOMERTest extends MultiLabelMetaLearnerTest {

    @Override
    public void setUp() throws Exception {
        learner = new HOMER(new LabelPowerset(new J48()), 3, HierarchyBuilder.Method.Random);
    }
    
}
