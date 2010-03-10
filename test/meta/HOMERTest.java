package meta;

import mulan.classifier.lazy.BRkNN;
import mulan.classifier.meta.HOMER;
import mulan.classifier.meta.HierarchyBuilder;

public class HOMERTest extends MultiLabelMetaLearnerTest {

	@Override
	public void setUp() throws Exception {
		learner = new HOMER(new BRkNN(10), 3, HierarchyBuilder.Method.Random);
	}

}
