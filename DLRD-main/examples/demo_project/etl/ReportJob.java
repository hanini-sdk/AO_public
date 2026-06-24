// Demo report job. Illustration only: the analyzer parses this statically and
// never executes it. Provides a class with two methods to populate class and
// method nodes in the graph.
public class ReportJob {
    public void run() {
        publish();
    }

    public void publish() {
        System.out.println("Report published.");
    }
}
