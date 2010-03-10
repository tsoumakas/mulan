package mulan.classifier.lazy;

import org.junit.Before;

public class IBLR_MLTest extends MultiLabelKNNTest {
	
	private static final int DEFAULT_numOfNeighbors = 10; 
		
	@Override
	public void setUp(){
		learner = new  IBLR_ML(DEFAULT_numOfNeighbors);
	}

}
