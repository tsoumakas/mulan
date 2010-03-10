package mulan.classifier.meta;

import mulan.classifier.lazy.BRkNN;

public class RAkELdTest extends MultiLabelMetaLearnerTest {

	@Override
	public void setUp() throws Exception {
		learner = new RAkELd(new BRkNN(10));
	}

}
