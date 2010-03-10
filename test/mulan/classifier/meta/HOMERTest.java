package mulan.classifier.meta;

import mulan.classifier.lazy.BRkNN;

public class HOMERTest extends MultiLabelMetaLearnerTest {

	@Override
	public void setUp() throws Exception {
		learner = new HOMER(new BRkNN(10), 3, HierarchyBuilder.Method.Random);
	}

}
