package mulan.classifier.lazy;


public class IBLR_MLTest extends MultiLabelKNNTest {
	
	private static final int DEFAULT_numOfNeighbors = 10; 
		
	@Override
	public void setUp(){
		learner = new  IBLR_ML(DEFAULT_numOfNeighbors);
	}

}
