import { Topic } from "./Topic";

export class Post {
    idPost: any;
    content: any;
    likes: any;
    dislikes: any;
    creationDate: Date;
    modified: any;
    userId: any;
    user: any;
    topic: Topic;
    comments: Comment[];
  }
