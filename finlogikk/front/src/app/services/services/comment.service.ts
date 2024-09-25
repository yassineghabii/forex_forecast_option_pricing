
import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable, tap } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class CommentService {
  private API_URL = "http://localhost:8088/api/v1/comment";
  private API_URL1 = "http://localhost:8088/api/v1/voteComment";
  constructor(private httpClient: HttpClient) { }

  getAllComments(){
    return this.httpClient.get(`${this.API_URL}/getAllComments`)
  }
  addComment(comment : any, IdUser : any , IdPost : number) {
    return this.httpClient.post(`${this.API_URL}/addComment/${IdUser}/${IdPost}`, comment)
  }
  updateComment(comment : any, IdComment : number){
    return  this.httpClient.put(`${this.API_URL}/updateComment/${IdComment}`, comment)
  }
  deleteComment(IdComment : number){
    return  this.httpClient.delete(`${this.API_URL}/deleteComment/${IdComment}`)
  }
  getComment(IdComment : any){
    return  this.httpClient.get(`${this.API_URL}/getComment/${IdComment}`)
  }

  getCommentsByPostId(IdPost : any){
    return this.httpClient.get(`${this.API_URL}/byPost/${IdPost}`)
  }

  voteLike(IdComment:any,IdUser:any){
    return this.httpClient.post(`${this.API_URL1}/voteLike/${IdComment}/${IdUser}`,null);
  }
  voteDislike(IdComment:any,IdUser:any){
    return this.httpClient.post(`${this.API_URL1}/voteDislike/${IdComment}/${IdUser}`,null);
  }
  getUserVoteStatus2(CommentId: any, userId: any): Observable<string> {
    const url = `${this.API_URL1}/status/${CommentId}/${userId}`;
    return this.httpClient.get(url, { responseType: 'text' });
  }

}
