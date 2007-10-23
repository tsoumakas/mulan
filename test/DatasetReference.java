import java.io.FileReader;
import java.io.Serializable;

import weka.core.Instances;


public class DatasetReference implements Serializable
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 7077560474270384251L;
	protected String baseDir;
	protected String trainFile;
	protected String testFile;
	protected int numLabels;

	public DatasetReference(String train, String test, int numLabels) {
		trainFile = train;
		testFile  = test;
		this.numLabels = numLabels;
	}

	/**
	 * Called from Experiment instance, injecting the proper baseDir
	 * matching the file system of the current researcher.
	 */
	protected void setBaseDir(String dir)
	{
		baseDir = dir;
	}
	
	public Instances getTrain()
	{
		return get(baseDir + trainFile);
	}

	/**
	 * The dataset used for testing. Ignored when performing
	 * crossvalidation. May be null. Don't call this multiple
	 * times because the file will be reread each time.
	 */
	public Instances getTest()
	{
		return get(baseDir + testFile);
	}	
	
	
	protected Instances get(String file)
	{
		try
		{
			return new Instances(new FileReader(file));	
		}
		catch(Exception e)
		{
			return null;
		}		
	}

	
	
}
