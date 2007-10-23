import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;


public class Log 
{
	private List<PrintStream> targets;
	private static Log instance;
	
	static
	{
		instance = new Log();
		
	}
	
	private Log()
	{
		targets = new ArrayList<PrintStream>();
	}
	
	public static void addTarget(PrintStream out)
	{
		instance.targets.add(out);
	}
	
	public static void log(String message)
	{
		if (instance.targets.size() == 0) return;
		
		String logmessage = new Date().toString() + " : " + message;
		for(PrintStream target: instance.targets)
		{
			target.println(logmessage);
			target.flush();
		}
		
	}

}
