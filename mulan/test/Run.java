import java.io.FileOutputStream;
import java.io.PrintStream;
import java.io.Serializable;
import java.util.Date;


/**
 * Recordings from a single run of an experiment.
 *
 */
public class Run implements Serializable
{

	/**
	 * 
	 */
	private static final long serialVersionUID = -5779974260717302491L;

	/**
	 * Whatever is returned by Version.Id when the 
	 * experiment is executing.
	 */
	public String libraryVersion;
	
	/**
	 * An arbitrary string set by researcher to identify a run.
	 */
	public String tag;
	
	/**
	 * 
	 */
	public Date startTime = new Date();
	
	/**
	 * First row contains column headers.
	 */
	public String[][] data;
	
	/**
	 * Dump info and data to file for import 
	 * into spreadsheet.
	 * @param file
	 */
	public void exportToCSV(String file) throws Exception
	{
		PrintStream out = new PrintStream(new FileOutputStream(file));
		for(int row = 0; row < data.length; row++)
		{
			out.print(data[row][0]);
			for(int col = 1; col < data[0].length; col++)
			{
				out.print(",");
				out.print(data[row][col]);
			}
			out.println();
		}
		
		out.close();
	}
	
	public void printTo(PrintStream out)
	{
		int maxLength = getMaxHeaderLength();
		
		for(int i = 1; i < data.length; i++)
		{
			out.println("------------------------------------------------------------");
			for(int j=0;j<data[0].length;j++)
			{
				int padLen = maxLength - data[0][j].length();
				String pad = "                                        ".substring(0, padLen);
				out.println(data[0][j] + pad + " : " + data[i][j]);
			}
		}
	}
	
	private int getMaxHeaderLength()
	{
		int max = 0;
		for(int i = 0; i < data[0].length; i++)
			if (data[0][i].length() > max) max = data[0][i].length();
		return max;
	}
	
}
