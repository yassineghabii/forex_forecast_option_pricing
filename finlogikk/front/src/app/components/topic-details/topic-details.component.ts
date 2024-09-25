import { Component, OnInit, ViewChild } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { PostService } from "../../services/services/post.service";
import { TopicService } from "../../services/services/topic.service";
import { CommentService } from "../../services/services/comment.service";
import { faThumbsUp, faThumbsDown, faTrashAlt, faEdit, faComment, faCommentAlt, faCommentDots, faComments } from '@fortawesome/free-regular-svg-icons'
import { faBars, faListAlt, faThumbsUp as faThumbsUpp, faThumbsDown as faThumbsDownn } from '@fortawesome/free-solid-svg-icons'
import { Router } from '@angular/router';
import { ElementRef } from '@angular/core';
import { WindowRef } from "../../services/services/window-ref.service";
import { forkJoin } from 'rxjs';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Post } from 'src/app/entities/Post';
import { KeycloakService } from "../../services/keycloak/keycloak.service";
import { Comment } from 'src/app/entities/Comment';
import * as bootstrap from 'bootstrap';
import { NgbModal } from '@ng-bootstrap/ng-bootstrap';
import { Topic } from 'src/app/entities/Topic';


@Component({
  selector: 'app-topic-details',
  templateUrl: './topic-details.component.html',
  styleUrls: ['./topic-details.component.scss']
})
export class TopicDetailsComponent implements OnInit {

  modalReference: ElementRef;

  faThumbsUp = faThumbsUp;
  faThumbsUpp = faThumbsUpp;
  faThumbsDown = faThumbsDown;
  faThumbsDownn = faThumbsDownn;
  faTrashAlt = faTrashAlt;
  faEdit = faEdit;
  faComments = faComments;
  faBars = faBars;
  faListAlt = faListAlt;

  topicId: number;
  postId: number;

  topicDetails: any;
  postDetails: any;

  posts: any;
  coments: any;
  visibleComments: any;

  Postupdate: any;
  Commentupdate: any;

  currentPage = 1;
  pageSize = 5;


  postForm: FormGroup;
  Post:Post;
  users: any;
  idUser: any;
  commentForm: FormGroup;
  Comment:Comment;

  userVoteStatus1: string;
  userVoteStatus2: string;


  constructor(
    private route: ActivatedRoute,
    private formBuilder: FormBuilder,
    private topicService: TopicService,
    private postService: PostService,
    private keycloakService: KeycloakService,
    private commentService: CommentService,
    private router: Router,
    private el: ElementRef,
    private windowRef: WindowRef,
    private modalService: NgbModal
  ) {}

  ngOnInit(): void {

    this.route.params.subscribe(params => {
      this.topicId = +params['id'];
      console.log('Topic ID:', this.topicId);


    });

    this.getTopicDetails();
    this.getPostsByTopic();
    this.getPostDetails();
    this.getCommentsByPost1();

    this.keycloakService.getAllUsers().subscribe(data => {
      this.users = data;
    });

    this.postForm = this.formBuilder.group({
      content: [''],
      user: [''],
      topic: ['']
    });

    this.Post = {
      idPost: null,
      content: null,
      likes: null,
      dislikes: null,
      creationDate: null,
      modified: null,
      userId: null,
      user: null,
      topic: null,
      comments: null
    }

    this.Postupdate = {
      idPost: null,
      content: null,
      likes: null,
      dislikes: null,
      creationDate: null,
      modified: null,
      user: null,
      topic: null,
      comments: null
    }

    this.commentForm = this.formBuilder.group({
      content: [''],
      user: [''],
      post: ['']
    });

    this.Comment = {
      idComment: null,
      content: null,
      likes: null,
      dislikes: null,
      creationDate: null,
      modified: null,
      userId: null,
      user: null,
      post: null
    }

    this.Commentupdate = {
      idComment: null,
      content: null,
      likes: null,
      dislikes: null,
      creationDate: null,
      modified: null,
      user: null,
      post: null
    }

  }

  isCurrentUserOwner(topic: Topic): boolean {
    const userId = this.getUserId();
    return userId !== null && topic.userId === userId;
  }

  isCurrentUserOwner1(post: Post): boolean {
    const userId = this.getUserId();
    return userId !== null && post.userId === userId;
  }

  isCurrentUserOwner2(comment: Comment): boolean {
    const userId = this.getUserId();
    return userId !== null && comment.userId === userId;
  }

  getTopicDetails() {
    this.topicService.getTopic(this.topicId).subscribe(res => {
      this.topicDetails = res;
    });
  }

  getPostDetails() {
    this.postService.getPost(this.postId).subscribe(res => {
      this.postDetails = res;
    });
  }

  getPostsByTopic() {
    this.postService.getPostsByTopicId(this.topicId).subscribe(res => {
      this.posts = res;
      console.log('Posts:', res);

      this.posts.forEach((post: any) => {

        this.postService.commentCount(post.idPost).subscribe((count: number) => {
          post.commentCount = count;
        });

        this.postService.getUserVoteStatus1(post.idPost, this.getUserId()).subscribe((userVoteStatus1: string) => {
          post.userVoteStatus1 = userVoteStatus1;
        });

      });
    });
  }

  getCommentsByPost1() {
    this.commentService.getCommentsByPostId(this.postId).subscribe(res => {
      this.coments = res;

      this.coments.forEach((comment: any) => {

        this.commentService.getUserVoteStatus2(comment.idComment, this.getUserId()).subscribe((userVoteStatus2: string) => {
          comment.userVoteStatus2 = userVoteStatus2;
        });

      });

    });
  }

  getCommentsByPost(postId: number) {
    this.commentService.getCommentsByPostId(postId).subscribe(res => {
      this.coments = res;

      this.coments.forEach((comment: any) => {

        this.commentService.getUserVoteStatus2(comment.idComment, this.getUserId()).subscribe((userVoteStatus2: string) => {
          comment.userVoteStatus2 = userVoteStatus2;
        });

      });

    });
  }




  deleteTopic(IdTopic:number){
    this.topicService.deleteTopic(IdTopic).subscribe(()=>{
      this.router.navigate(['/topic']);
    });
  }

  confirmerSuppression2(idTopic: number) {
    const estConfirme = this.windowRef.nativeWindow.confirm("Are you sure you want to delete this topic?");
    if (estConfirme) {
      this.deleteTopic(idTopic);
    }
  }




  votelikeP(idPost: any) {

    const userId = this.getUserId();

    this.postService.voteLike(idPost, userId).subscribe(
      (resp) => {
        console.log('Post liked successfully:', resp);
        this.getPostsByTopic();
      },
      (err) => {
        console.error('Error while liking Post:', err);
      }
    );
  }

  votedislikeP(idPost: any) {

    const userId = this.getUserId();

    this.postService.voteDislike(idPost, userId).subscribe(
      (resp) => {
        console.log('Post disliked successfully:', resp);
        this.getPostsByTopic();
      },
      (err) => {
        console.error('Error while disliking Post:', err);
      }
    );
  }


  votelikeC(idComment: any) {

    const userId = this.getUserId();

    this.commentService.voteLike(idComment, userId).subscribe(
      (resp) => {
        console.log('Comment liked successfully:', resp);
        this.getCommentsByPost1();
      },
      (err) => {
        console.error('Error while liking Comment:', err);
      }
    );
  }

  votedislikeC(idComment: any) {

    const userId = this.getUserId();

    this.commentService.voteDislike(idComment, userId).subscribe(
      (resp) => {
        console.log('Comment disliked successfully:', resp);
        this.getCommentsByPost1();
      },
      (err) => {
        console.error('Error while disliking Comment:', err);
      }
    );
  }


  addPost() {
    if (this.postForm.invalid) {
      console.log('Invalid form');
      return;
    }

    const userId = this.getUserId();
    const topicId = this.topicId;

    this.postService.addPost(this.postForm.value, userId, topicId).subscribe(
      (resp) => {
        console.log('Post added successfully:', resp);
        this.modalService.dismissAll();
        this.getPostsByTopic();
      },
      (err) => {
        console.log('Error while adding post:', err);
      }
    );
  }

  addComment() {
    if (this.commentForm.invalid) {
      console.log('Invalid form');
      return;
    }

    const userId = this.getUserId();
    const postId = this.postId;

    this.commentService.addComment(this.commentForm.value, userId, postId).subscribe(
      (resp) => {
        console.log('Comment added successfully:', resp);
        this.getCommentsByPost1();
        // window.location.reload();
      },
      (err) => {
        console.log('Error while adding comment:', err);
      }
    );
  }

  edit(post: any, content: any){
    this.Postupdate = post;

    this.modalService.open(content, { ariaLabelledBy: 'commentModalLabel' });
  }

  add(post: any){
    this.Post = post;
  }

  edit1(comment: any, content: any){
    this.Commentupdate = comment;

    this.modalService.open(content, { ariaLabelledBy: 'commentModalLabel' });
  }

  add1(comment: any){
    this.Comment = comment;
  }

  updatePost(){
    this.postService.updatePost(this.Postupdate, this.Postupdate.idPost).subscribe(
      (resp) => {
        console.log('Post updated successfully:', resp);
        this.modalService.dismissAll();
        this.getPostsByTopic();
      },
      (err) => {
        console.log('Error while updating post:', err);
      }
    );}

  updateComment(){
    this.commentService.updateComment(this.Commentupdate, this.Commentupdate.idComment).subscribe(
      (resp) => {
        console.log('Comment updated successfully:', resp);
        this.getCommentsByPost1();
        // window.location.reload();
      },
      (err) => {
        console.log('Error while updating comment:', err);
      }
    );}

  deletePost(IdPost:number){
    this.postService.deletePost(IdPost).subscribe(()=>{
      this.modalService.dismissAll();
      this.getPostsByTopic();

    }, (err) => {
      console.error('Error while deleting post:', err);
    });
  }

  deleteComment(IdComment:number){
    this.commentService.deleteComment(IdComment).subscribe(()=>{
      this.getCommentsByPost1();
      // window.location.reload();
    }, (err) => {
      console.error('Error while deleting comment:', err);
    });
  }


  confirmerSuppression(idPost: number) {
    const estConfirme = this.windowRef.nativeWindow.confirm("Are you sure you want to delete this post?");
    if (estConfirme) {
      this.deletePost(idPost);
    }
  }

  confirmerSuppression1(IdComment: number) {
    const estConfirme = this.windowRef.nativeWindow.confirm("Are you sure you want to delete this coment?");
    if (estConfirme) {
      this.deleteComment(IdComment);
    }
  }


  getUserId(): string | null {
    const tokenParsed = this.keycloakService.keycloak.tokenParsed; // Keycloak décode automatiquement le token JWT
    const userId = tokenParsed ? tokenParsed.sub : null; // 'sub' est l'ID utilisateur
    return userId;
  }

  openModal1(content: any) {
    this.modalService.open(content, { ariaLabelledBy: 'modal-basic-title' });
  }
  openModal2(content: any) {
    this.modalService.open(content, { ariaLabelledBy: 'modal-basic-title' });
  }


  openModal4(content: any) {
    this.modalService.open(content, { ariaLabelledBy: 'modal-basic-title' });
  }

  openModal6(content: any) {
    this.modalService.open(content, { ariaLabelledBy: 'modal-basic-title' });
  }


  openCommentModal(content: any) {
    this.modalService.open(content, { ariaLabelledBy: 'modal-basic-title' });
  }


  showCommentsModal(post: any, content: any) {
    console.log(post.idPost);
    if (post.idPost) {
      this.postId = post.idPost; // Assurez-vous de définir postId ici
      this.getCommentsByPost(post.idPost);

      this.postService.commentCount(post.idPost).subscribe((count: number) => {
        this.visibleComments = count;
      });

      this.modalService.open(content, { ariaLabelledBy: 'commentModalLabel' });
    }
  }


}
