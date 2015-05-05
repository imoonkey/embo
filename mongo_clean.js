conn = new Mongo();
db = conn.getDB("spearmint");
db['simple-bo-hmm.jobs'].drop();
db['simple-bo-hmm.hypers'].drop();