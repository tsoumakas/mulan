import mulan.classifier.transformation.*;

public class runningTest {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		COCOATest t=new COCOATest();
		//BinaryRelevanceTest t=new BinaryRelevanceTest();
		//BinaryRelevanceUnderSamplingTest t=new BinaryRelevanceUnderSamplingTest();
		t.setUp();
		try{
			System.out.println("testBuild"); t.testBuild();
			System.out.println("testBuild_WithDifferentOrder"); t.testBuild_WithDifferentOrder();
			System.out.println("testBuild_WithMissingValues"); t.testBuild_WithMissingValues();  //weka.filters.supervised.instance.SpreadSubsample: Cannot handle missing class values!
			
			System.out.println("testBuildWith10ClassLabels"); t.testBuildWith10ClassLabels();
			System.out.println("testBuildWithNonSparse"); t.testBuildWithNonSparse();
			System.out.println("testGetTechnicalInformation"); t.testGetTechnicalInformation();
			System.out.println("testMakeCopy"); t.testMakeCopy();
			System.out.println("testMakePrediction_Generic"); t.testMakePrediction_Generic();
			System.out.println("testSameInputSameOutput"); t.testSameInputSameOutput();
			
		}
		catch (Exception e){
			e.printStackTrace();
		}
		
		
		try{
			System.out.println("testBuild_WithNullDataSet"); t.testBuild_WithNullDataSet();  
		}
		catch (Exception e){
			System.out.println("testBuild_WithNullDataSet is passed");
			e.printStackTrace();
		}
		try{
			System.out.println("testMakePrediction_BeforeBuild"); t.testMakePrediction_BeforeBuild();  
		}
		catch (Exception e){
			System.out.println("testMakePrediction_BeforeBuild is passed");
			e.printStackTrace();
		}
		try{
			System.out.println("testMakePrediction_WithNullData"); t.testMakePrediction_WithNullData();  
		}
		catch (Exception e){
			System.out.println("testMakePrediction_WithNullData is passed");
			e.printStackTrace();
		}
		
	}

}
