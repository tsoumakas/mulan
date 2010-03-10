package mulan.classifier.meta;

import mulan.classifier.lazy.BRkNN;

public class RAkELTest extends MultiLabelMetaLearnerTest {

	@Override
	public void setUp() throws Exception {
		learner = new RAkEL(new BRkNN(10));
	}

}
