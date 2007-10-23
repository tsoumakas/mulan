import java.io.PrintStream;
import java.util.Vector;


public class PerformanceTrace
{
	private Vector<Tuple> timings;
	private long start;
	private boolean autoPrint;
	
	public void trace(String description)
	{
		Tuple tuple = new Tuple();
		tuple.description = description;
		tuple.nanosFromStart = System.nanoTime();
		timings.add(tuple);
		if (autoPrint) printTuple(tuple, System.err);
	}
	
	public PerformanceTrace(boolean print)
	{
		timings = new Vector<Tuple>();
		start = System.nanoTime();
		autoPrint = print;
	}
	
	public void print(PrintStream out)
	{
		for(Tuple tuple : timings) printTuple(tuple, out);
	}
	
	private void printTuple(Tuple tuple, PrintStream out)
	{
		double ms = ((double)(tuple.nanosFromStart - start)) / 1000000.0;
		out.println(ms + " | " + tuple.description );
	}
	
	private class Tuple
	{
		public long nanosFromStart;
		public String description;
	}
}
