/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */

package mulan.console.driver;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.neural.BPMLL;
import mulan.data.MultiLabelInstances;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.OptionGroup;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

public abstract class LearnerDriverBase implements LearnerDriver {

	private final String HELP_OPT = "h";
	private final String TRAIN_DATA_OPT = "trd";
	private final String TEST_DATA_OPT = "tsd";
	protected final String SEED_OPT = "s";
	
	@SuppressWarnings("static-access")
	@Override
	public Options getOptions() {
		Options opt = new Options();
		
		opt.addOption(OptionBuilder.withLongOpt("train-data").hasArg().withType(File.class).create(TRAIN_DATA_OPT));
		opt.addOption(OptionBuilder.withLongOpt("test-data").hasArg().withType(File.class).create(TEST_DATA_OPT));
		opt.addOption(OptionBuilder.withLongOpt("seed").hasArg().withType(Number.class).create(SEED_OPT));
		opt.addOption(OptionBuilder.withLongOpt("help").create(HELP_OPT));
		
		return opt;
	}
	
	@Override
	public final void run(String[] args) {
		CommandLineParser parser = new GnuParser();
		try {
	        // parse the command line arguments
	        CommandLine cmdLine = parser.parse(getOptions(), args);
	        
	        if(cmdLine.hasOption(HELP_OPT)){
	        	HelpFormatter formatter = new HelpFormatter();
	        	formatter.printHelp("command", getOptions(), true);
	        } else{
	        	String train = cmdLine.getOptionValue(TRAIN_DATA_OPT);
	        	String test = cmdLine.getOptionValue(TEST_DATA_OPT);
	        	
	        	MultiLabelLearner learner = createLearner(cmdLine);

	        	// TODO: build learner and evaluate
	        }
	        
	    }
	    catch( ParseException exp ) {
	        System.err.println( "Parsing failed.  Reason: " + exp.getMessage() );
	    }
		
	}
	
	protected abstract MultiLabelLearner createLearner(CommandLine cmdLine);

}
